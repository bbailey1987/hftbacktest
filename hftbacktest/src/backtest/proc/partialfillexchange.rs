use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::{HashMap, HashSet},
    num::NonZeroU64,
    rc::Rc,
};

use crate::{
    backtest::{
        BacktestError,
        assettype::AssetType,
        models::{FeeModel, LatencyModel, QueueModel},
        order::ExchToLocal,
        proc::Processor,
        state::State,
    },
    depth::{INVALID_MAX, INVALID_MIN, L2MarketDepth, MarketDepth},
    prelude::OrdType,
    types::{
        EXCH_ASK_DEPTH_CLEAR_EVENT, EXCH_ASK_DEPTH_EVENT, EXCH_ASK_DEPTH_SNAPSHOT_EVENT,
        EXCH_BID_DEPTH_CLEAR_EVENT, EXCH_BID_DEPTH_EVENT, EXCH_BID_DEPTH_SNAPSHOT_EVENT,
        EXCH_BUY_TRADE_EVENT, EXCH_DEPTH_CLEAR_EVENT, EXCH_EVENT, EXCH_SELL_TRADE_EVENT, Event,
        Order, OrderId, Side, Status, TimeInForce,
    },
};

/// Identity of a submitted batch, independent of timestamps. Created by its registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct IocBatchId(NonZeroU64);

#[derive(Default, Debug)]
struct IocBatchRegistryState {
    next_batch: u64,
    orders: HashMap<OrderId, IocBatchId>,
}

/// Shared only with the submitting thread of one exchange. Register accepted IOC
/// orders before advancing its clock. The order bus must deliver batches in
/// submission order, with each batch contiguous. Entries are removed at arrival.
#[derive(Clone, Default, Debug)]
pub struct IocBatchRegistry(Rc<RefCell<IocBatchRegistryState>>);

/// An order cannot belong to two in-flight batches.
#[derive(Debug)]
pub struct DuplicateIocBatchOrder(pub OrderId);

impl IocBatchRegistry {
    pub fn begin_batch(&self) -> IocBatchId {
        let mut state = self.0.borrow_mut();
        state.next_batch = state
            .next_batch
            .checked_add(1)
            .expect("IOC submitted batch identity must not overflow");
        IocBatchId(
            NonZeroU64::new(state.next_batch).expect("IOC submitted batch identity starts at one"),
        )
    }

    pub fn register_order(
        &self,
        order_id: OrderId,
        batch: IocBatchId,
    ) -> Result<(), DuplicateIocBatchOrder> {
        use std::collections::hash_map::Entry;
        match self.0.borrow_mut().orders.entry(order_id) {
            Entry::Vacant(entry) => {
                entry.insert(batch);
                Ok(())
            }
            Entry::Occupied(_) => Err(DuplicateIocBatchOrder(order_id)),
        }
    }

    fn receive(&self, order_id: OrderId) -> IocBatchId {
        self.0.borrow_mut().orders.remove(&order_id).expect(
            "every accepted IOC must have its submitted batch registered before exchange arrival",
        )
    }
}

/// Applies a feed update of one price level to what our own taker fills took from it. `level_qty` is the quantity
/// the depth holds at the level after the update (zero once the depth has dropped the level). Without `persists`
/// the feed update makes the level whole again. With it, what we took stays taken, capped at the quantity the level
/// now holds, until the level is empty.
fn own_fills_after_level_update(
    consumed: &mut HashMap<i64, f64>,
    persists: bool,
    price_tick: i64,
    level_qty: f64,
) {
    if !persists || level_qty <= 0.0 {
        consumed.remove(&price_tick);
    } else if let Some(taken) = consumed.get_mut(&price_tick) {
        if *taken > level_qty {
            *taken = level_qty;
        }
    }
}

/// The quantity of a level a taker order can still fill against, given what our own fills took from it. With
/// persisting depletion a level can stay fully taken across many orders, so a remainder below half a lot, which is
/// only the rounding left by adding our fills up, is nothing rather than a fill of that size.
fn own_fill_available(level_qty: f64, taken: f64, persists: bool, lot_size: f64) -> f64 {
    let available = level_qty - taken;
    if persists && available < 0.5 * lot_size {
        0.0
    } else {
        available
    }
}

/// The exchange model with partial fills.
///
/// * Support order types: [OrdType::Limit](crate::types::OrdType::Limit)
/// * Support time-in-force: [`TimeInForce::GTC`], [`TimeInForce::FOK`], [`TimeInForce::IOC`],
///   [`TimeInForce::GTX`]
///
/// **Conditions for Full Execution**
/// Buy order in the order book
///
/// - Your order price >= the best ask price
/// - Your order price > sell trade price
///
/// Sell order in the order book
///
/// - Your order price <= the best bid price
/// - Your order price < buy trade price
///
/// **Conditions for Partial Execution**
/// Buy order in the order book
///
/// - Filled by (remaining) sell trade quantity: your order is at the front of the queue && your
///   order price == sell trade price
///
/// Sell order in the order book
///
/// - Filled by (remaining) buy trade quantity: your order is at the front of the queue && your
///   order price == buy trade price
///
/// **Liquidity-Taking Order**
/// Liquidity-taking orders will be executed based on the quantity of the order book, even though
/// the best price and quantity do not change due to your execution. Be aware that this may cause
/// unrealistic fill simulations if you attempt to execute a large quantity.
///
/// **General Comment**
/// Simulating partial fills accurately can be challenging, as they may indicate potential market
/// impact. The rule of thumb is to ensure that your backtesting results align with your live
/// results.
/// (more comment will be added...)
///
pub struct PartialFillExchange<AT, LM, QM, MD, FM>
where
    AT: AssetType,
    LM: LatencyModel,
    QM: QueueModel<MD>,
    MD: MarketDepth,
    FM: FeeModel,
{
    // key: order_id, value: Order
    orders: Rc<RefCell<HashMap<OrderId, Order>>>,
    // key: order's price tick, value: order_ids
    buy_orders: HashMap<i64, HashSet<OrderId>>,
    sell_orders: HashMap<i64, HashSet<OrderId>>,

    order_e2l: ExchToLocal<LM>,

    depth: MD,
    state: State<AT, FM>,
    queue_model: QM,

    filled_orders: Vec<OrderId>,

    // Own-fill depletion (opt-in): the quantity our own IOC taker fills took at each price level, since that level's
    // last feed update or, with `own_fill_depletion_persists`, since the level was last empty. Replayed depth never
    // reflects our own fills, so without it a later taker order can fill again against displayed shares one of
    // ours already took. A depth clear event forgets the whole side.
    own_fill_depletion: bool,
    // With own-fill depletion: `false` (the default) forgets what we took from a level when the feed next updates
    // that level; `true` keeps it across feed updates until the feed empties the level.
    own_fill_depletion_persists: bool,
    ioc_batches: Option<IocBatchRegistry>,
    active_ioc_batch: Option<IocBatchId>,
    consumed_bid: HashMap<i64, f64>,
    consumed_ask: HashMap<i64, f64>,
}

impl<AT, LM, QM, MD, FM> PartialFillExchange<AT, LM, QM, MD, FM>
where
    AT: AssetType,
    LM: LatencyModel,
    QM: QueueModel<MD>,
    MD: MarketDepth,
    FM: FeeModel,
{
    /// Constructs an instance of `PartialFillExchange`.
    pub fn new(
        depth: MD,
        state: State<AT, FM>,
        queue_model: QM,
        order_e2l: ExchToLocal<LM>,
    ) -> Self {
        Self {
            orders: Default::default(),
            buy_orders: Default::default(),
            sell_orders: Default::default(),
            order_e2l,
            depth,
            state,
            queue_model,
            filled_orders: Default::default(),
            own_fill_depletion: false,
            own_fill_depletion_persists: false,
            ioc_batches: None,
            active_ioc_batch: None,
            consumed_bid: Default::default(),
            consumed_ask: Default::default(),
        }
    }

    /// Opt-in own-fill depletion: an IOC taker order cannot fill against displayed quantity that our own earlier IOC
    /// taker fills already took at that price level, until the feed next updates the level (or, with
    /// [`Self::with_own_fill_depletion_persists`], until the level is empty).
    pub fn with_own_fill_depletion(mut self, enabled: bool) -> Self {
        self.ioc_batches
            .is_none()
            .then_some(())
            .expect("submitted-batch matching cannot disable its shared liquidity budget");
        self.own_fill_depletion = enabled;
        self
    }

    /// With own-fill depletion enabled, what our own IOC taker fills took from a price level stays taken across feed
    /// updates of that level instead of being forgotten at the next one. An update that leaves less than we took
    /// caps what we took at the new displayed quantity, quantity displayed beyond what we took is available, and a
    /// level the feed empties starts fresh. Across level updates this is the conservative bound: it assumes an
    /// update never replaces what we took, where the default assumes it always does. A depth clear event still
    /// forgets the whole side, and only IOC orders read or add to what was taken.
    pub fn with_own_fill_depletion_persists(mut self, enabled: bool) -> Self {
        self.ioc_batches.is_none().then_some(())
            .expect("submitted-batch matching cannot retain deductions between batches or level replacements");
        self.own_fill_depletion_persists = enabled;
        self
    }

    /// Mandatory shared liquidity within each explicit submitted IOC batch, with
    /// no deduction carried into the next batch. Clear at exchange arrival, not
    /// registration, so several batches may be in flight during entry latency.
    /// Authoritative feed updates still replace level quantity normally.
    pub fn with_ioc_batches(mut self, batches: IocBatchRegistry) -> Self {
        self.ioc_batches = Some(batches);
        self.own_fill_depletion = true;
        self.own_fill_depletion_persists = false;
        self
    }

    fn check_if_sell_filled(
        &mut self,
        order: &mut Order,
        price_tick: i64,
        qty: f64,
        timestamp: i64,
    ) -> Result<(), BacktestError> {
        match order.price_tick.cmp(&price_tick) {
            Ordering::Greater => {}
            Ordering::Less => {
                self.filled_orders.push(order.order_id);
                return self.fill::<true>(
                    order,
                    timestamp,
                    true,
                    order.price_tick,
                    order.leaves_qty,
                );
            }
            Ordering::Equal => {
                // Updates the order's queue position.
                self.queue_model.trade(order, qty, &self.depth);
                let filled_qty = self.queue_model.is_filled(order, &self.depth);
                if filled_qty > 0.0 {
                    // q_ahead is negative since is_filled is true and its value represents the
                    // executable quantity of this order after execution in the queue ahead of this
                    // order.
                    let exec_qty = if filled_qty > order.leaves_qty {
                        self.filled_orders.push(order.order_id);
                        order.leaves_qty
                    } else {
                        filled_qty
                    };
                    return self.fill::<true>(order, timestamp, true, order.price_tick, exec_qty);
                }
            }
        }
        Ok(())
    }

    fn check_if_buy_filled(
        &mut self,
        order: &mut Order,
        price_tick: i64,
        qty: f64,
        timestamp: i64,
    ) -> Result<(), BacktestError> {
        match order.price_tick.cmp(&price_tick) {
            Ordering::Greater => {
                self.filled_orders.push(order.order_id);
                return self.fill::<true>(
                    order,
                    timestamp,
                    true,
                    order.price_tick,
                    order.leaves_qty,
                );
            }
            Ordering::Less => {}
            Ordering::Equal => {
                // Updates the order's queue position.
                self.queue_model.trade(order, qty, &self.depth);
                let filled_qty = self.queue_model.is_filled(order, &self.depth);
                if filled_qty > 0.0 {
                    // q_ahead is negative since is_filled is true and its value represents the
                    // executable quantity of this order after execution in the queue ahead of this
                    // order.
                    let exec_qty = if filled_qty > order.leaves_qty {
                        self.filled_orders.push(order.order_id);
                        order.leaves_qty
                    } else {
                        filled_qty
                    };
                    return self.fill::<true>(order, timestamp, true, order.price_tick, exec_qty);
                }
            }
        }
        Ok(())
    }

    fn fill<const MAKE_RESPONSE: bool>(
        &mut self,
        order: &mut Order,
        timestamp: i64,
        maker: bool,
        exec_price_tick: i64,
        exec_qty: f64,
    ) -> Result<(), BacktestError> {
        if order.status == Status::Expired
            || order.status == Status::Canceled
            || order.status == Status::Filled
        {
            return Err(BacktestError::InvalidOrderStatus);
        }

        order.maker = maker;
        if maker {
            order.exec_price_tick = order.price_tick;
        } else {
            order.exec_price_tick = exec_price_tick;
        }

        order.exec_qty = exec_qty;
        order.leaves_qty -= exec_qty;
        if (order.leaves_qty / self.depth.lot_size()).round() > 0f64 {
            order.status = Status::PartiallyFilled;
        } else {
            order.status = Status::Filled;
        }
        order.exch_timestamp = timestamp;

        self.state.apply_fill(order);

        if MAKE_RESPONSE {
            self.order_e2l.respond(order.clone());
        }
        Ok(())
    }

    fn remove_filled_orders(&mut self) {
        if !self.filled_orders.is_empty() {
            let mut orders = self.orders.borrow_mut();
            for order_id in self.filled_orders.drain(..) {
                let order = orders.remove(&order_id).unwrap();
                if order.side == Side::Buy {
                    self.buy_orders
                        .get_mut(&order.price_tick)
                        .unwrap()
                        .remove(&order_id);
                } else {
                    self.sell_orders
                        .get_mut(&order.price_tick)
                        .unwrap()
                        .remove(&order_id);
                }
            }
        }
    }

    fn on_bid_qty_chg(&mut self, price_tick: i64, prev_qty: f64, new_qty: f64) {
        let orders = self.orders.clone();
        if let Some(order_ids) = self.buy_orders.get(&price_tick) {
            for order_id in order_ids.iter() {
                let mut orders_borrowed = orders.borrow_mut();
                let order = orders_borrowed.get_mut(order_id).unwrap();
                self.queue_model
                    .depth(order, prev_qty, new_qty, &self.depth);
            }
        }
    }

    fn on_ask_qty_chg(&mut self, price_tick: i64, prev_qty: f64, new_qty: f64) {
        let orders = self.orders.clone();
        if let Some(order_ids) = self.sell_orders.get(&price_tick) {
            for order_id in order_ids.iter() {
                let mut orders_borrowed = orders.borrow_mut();
                let order = orders_borrowed.get_mut(order_id).unwrap();
                self.queue_model
                    .depth(order, prev_qty, new_qty, &self.depth);
            }
        }
    }

    fn on_best_bid_update(
        &mut self,
        prev_best_tick: i64,
        new_best_tick: i64,
        timestamp: i64,
    ) -> Result<(), BacktestError> {
        // If the best has been significantly updated compared to the previous best, it would be
        // better to iterate orders dict instead of order price ladder.
        {
            let orders = self.orders.clone();
            let mut orders_borrowed = orders.borrow_mut();
            if prev_best_tick == INVALID_MIN
                || (orders_borrowed.len() as i64) < new_best_tick - prev_best_tick
            {
                for (_, order) in orders_borrowed.iter_mut() {
                    if order.side == Side::Sell && order.price_tick <= new_best_tick {
                        self.filled_orders.push(order.order_id);
                        self.fill::<true>(
                            order,
                            timestamp,
                            true,
                            order.price_tick,
                            order.leaves_qty,
                        )?;
                    }
                }
            } else {
                for t in (prev_best_tick + 1)..=new_best_tick {
                    if let Some(order_ids) = self.sell_orders.get(&t) {
                        for order_id in order_ids.clone().iter() {
                            self.filled_orders.push(*order_id);
                            let order = orders_borrowed.get_mut(order_id).unwrap();
                            self.fill::<true>(
                                order,
                                timestamp,
                                true,
                                order.price_tick,
                                order.leaves_qty,
                            )?;
                        }
                    }
                }
            }
        }
        self.remove_filled_orders();
        Ok(())
    }

    fn on_best_ask_update(
        &mut self,
        prev_best_tick: i64,
        new_best_tick: i64,
        timestamp: i64,
    ) -> Result<(), BacktestError> {
        // If the best has been significantly updated compared to the previous best, it would be
        // better to iterate orders dict instead of order price ladder.
        {
            let orders = self.orders.clone();
            let mut orders_borrowed = orders.borrow_mut();
            if prev_best_tick == INVALID_MAX
                || (orders_borrowed.len() as i64) < prev_best_tick - new_best_tick
            {
                for (_, order) in orders_borrowed.iter_mut() {
                    if order.side == Side::Buy && order.price_tick >= new_best_tick {
                        self.filled_orders.push(order.order_id);
                        self.fill::<true>(
                            order,
                            timestamp,
                            true,
                            order.price_tick,
                            order.leaves_qty,
                        )?;
                    }
                }
            } else {
                for t in new_best_tick..prev_best_tick {
                    if let Some(order_ids) = self.buy_orders.get(&t) {
                        for order_id in order_ids.clone().iter() {
                            self.filled_orders.push(*order_id);
                            let order = orders_borrowed.get_mut(order_id).unwrap();
                            self.fill::<true>(
                                order,
                                timestamp,
                                true,
                                order.price_tick,
                                order.leaves_qty,
                            )?;
                        }
                    }
                }
            }
        }
        self.remove_filled_orders();
        Ok(())
    }

    // AB#3112 (gentilly): every level a taker walks is reported in its own
    // response (`fill::<true>`), so the local processor can book each fill at
    // its own price; the final ack response then reports no new execution.
    fn ack_new(&mut self, order: &mut Order, timestamp: i64) -> Result<(), BacktestError> {
        if self.orders.borrow().contains_key(&order.order_id) {
            return Err(BacktestError::OrderIdExist);
        }

        if order.side == Side::Buy {
            match order.order_type {
                OrdType::Limit => {
                    // Checks if the buy order price is greater than or equal to the current best ask.
                    if order.price_tick >= self.depth.best_ask_tick() {
                        match order.time_in_force {
                            TimeInForce::GTX => {
                                order.status = Status::Expired;
                                order.exch_timestamp = timestamp;
                                Ok(())
                            }
                            TimeInForce::FOK => {
                                // The order must be executed immediately in its entirety; otherwise, the
                                // entire order will be cancelled.
                                let mut execute = false;
                                let mut cum_qty = 0f64;
                                for t in self.depth.best_ask_tick()..=order.price_tick {
                                    cum_qty += self.depth.ask_qty_at_tick(t);
                                    if (cum_qty / self.depth.lot_size()).round()
                                        >= (order.qty / self.depth.lot_size()).round()
                                    {
                                        execute = true;
                                        break;
                                    }
                                }
                                if execute {
                                    for t in self.depth.best_ask_tick()..=order.price_tick {
                                        let qty = self.depth.ask_qty_at_tick(t);
                                        if qty > 0.0 {
                                            let exec_qty = qty.min(order.leaves_qty);
                                            self.fill::<true>(
                                                order, timestamp, false, t, exec_qty,
                                            )?;
                                            if order.status == Status::Filled {
                                                return Ok(());
                                            }
                                        }
                                    }
                                    unreachable!();
                                } else {
                                    order.status = Status::Expired;
                                    order.exch_timestamp = timestamp;
                                    Ok(())
                                }
                            }
                            TimeInForce::IOC => {
                                // The order must be executed immediately.
                                for t in self.depth.best_ask_tick()..=order.price_tick {
                                    let mut qty = self.depth.ask_qty_at_tick(t);
                                    if self.own_fill_depletion {
                                        qty = own_fill_available(
                                            qty,
                                            self.consumed_ask.get(&t).copied().unwrap_or(0.0),
                                            self.own_fill_depletion_persists,
                                            self.depth.lot_size(),
                                        );
                                    }
                                    if qty > 0.0 {
                                        let exec_qty = qty.min(order.leaves_qty);
                                        if self.own_fill_depletion {
                                            *self.consumed_ask.entry(t).or_insert(0.0) += exec_qty;
                                        }
                                        self.fill::<true>(order, timestamp, false, t, exec_qty)?;
                                    }
                                    if order.status == Status::Filled {
                                        return Ok(());
                                    }
                                }
                                order.status = Status::Expired;
                                order.exch_timestamp = timestamp;
                                Ok(())
                            }
                            TimeInForce::GTC => {
                                // Takes the market.
                                for t in self.depth.best_ask_tick()..order.price_tick {
                                    let qty = self.depth.ask_qty_at_tick(t);
                                    if qty > 0.0 {
                                        let exec_qty = qty.min(order.leaves_qty);
                                        self.fill::<true>(order, timestamp, false, t, exec_qty)?;
                                    }
                                    if order.status == Status::Filled {
                                        return Ok(());
                                    }
                                }

                                // The buy order cannot remain in the ask book, as it cannot affect the
                                // market depth during backtesting based on market-data replay. So, even
                                // though it simulates partial fill, if the order size is not small enough,
                                // it introduces unreality.
                                let (price_tick, leaves_qty) = (order.price_tick, order.leaves_qty);
                                self.fill::<true>(order, timestamp, false, price_tick, leaves_qty)
                            }
                            TimeInForce::Unsupported => Err(BacktestError::InvalidOrderRequest),
                        }
                    } else {
                        match order.time_in_force {
                            TimeInForce::GTC | TimeInForce::GTX => {
                                // Initializes the order's queue position.
                                self.queue_model.new_order(order, &self.depth);
                                order.status = Status::New;
                                // The exchange accepts this order.
                                self.buy_orders
                                    .entry(order.price_tick)
                                    .or_default()
                                    .insert(order.order_id);

                                order.exch_timestamp = timestamp;
                                self.orders
                                    .borrow_mut()
                                    .insert(order.order_id, order.clone());
                                Ok(())
                            }
                            TimeInForce::FOK | TimeInForce::IOC => {
                                order.status = Status::Expired;
                                order.exch_timestamp = timestamp;
                                Ok(())
                            }
                            TimeInForce::Unsupported => Err(BacktestError::InvalidOrderRequest),
                        }
                    }
                }
                OrdType::Market => {
                    // todo: set the proper upper bound.
                    for t in self.depth.best_ask_tick()..(self.depth.best_ask_tick() + 100) {
                        let qty = self.depth.ask_qty_at_tick(t);
                        if qty > 0.0 {
                            let exec_qty = qty.min(order.leaves_qty);
                            self.fill::<true>(order, timestamp, false, t, exec_qty)?;
                        }
                        if order.status == Status::Filled {
                            return Ok(());
                        }
                    }
                    order.status = Status::Expired;
                    order.exch_timestamp = timestamp;
                    Ok(())
                }
                OrdType::Unsupported => Err(BacktestError::InvalidOrderRequest),
            }
        } else {
            match order.order_type {
                OrdType::Limit => {
                    // Checks if the sell order price is less than or equal to the current best bid.
                    if order.price_tick <= self.depth.best_bid_tick() {
                        match order.time_in_force {
                            TimeInForce::GTX => {
                                order.status = Status::Expired;
                                order.exch_timestamp = timestamp;
                                Ok(())
                            }
                            TimeInForce::FOK => {
                                // The order must be executed immediately in its entirety; otherwise, the
                                // entire order will be cancelled.
                                let mut execute = false;
                                let mut cum_qty = 0f64;
                                for t in (order.price_tick..=self.depth.best_bid_tick()).rev() {
                                    cum_qty += self.depth.bid_qty_at_tick(t);
                                    if (cum_qty / self.depth.lot_size()).round()
                                        >= (order.qty / self.depth.lot_size()).round()
                                    {
                                        execute = true;
                                        break;
                                    }
                                }
                                if execute {
                                    for t in (order.price_tick..=self.depth.best_bid_tick()).rev() {
                                        let qty = self.depth.bid_qty_at_tick(t);
                                        if qty > 0.0 {
                                            let exec_qty = qty.min(order.leaves_qty);
                                            self.fill::<true>(
                                                order, timestamp, false, t, exec_qty,
                                            )?;
                                            if order.status == Status::Filled {
                                                return Ok(());
                                            }
                                        }
                                    }
                                    unreachable!();
                                } else {
                                    order.status = Status::Expired;
                                    order.exch_timestamp = timestamp;
                                    Ok(())
                                }
                            }
                            TimeInForce::IOC => {
                                // The order must be executed immediately.
                                for t in (order.price_tick..=self.depth.best_bid_tick()).rev() {
                                    let mut qty = self.depth.bid_qty_at_tick(t);
                                    if self.own_fill_depletion {
                                        qty = own_fill_available(
                                            qty,
                                            self.consumed_bid.get(&t).copied().unwrap_or(0.0),
                                            self.own_fill_depletion_persists,
                                            self.depth.lot_size(),
                                        );
                                    }
                                    if qty > 0.0 {
                                        let exec_qty = qty.min(order.leaves_qty);
                                        if self.own_fill_depletion {
                                            *self.consumed_bid.entry(t).or_insert(0.0) += exec_qty;
                                        }
                                        self.fill::<true>(order, timestamp, false, t, exec_qty)?;
                                    }
                                    if order.status == Status::Filled {
                                        return Ok(());
                                    }
                                }
                                order.status = Status::Expired;
                                order.exch_timestamp = timestamp;
                                Ok(())
                            }
                            TimeInForce::GTC => {
                                // Takes the market.
                                for t in (order.price_tick..=self.depth.best_bid_tick()).rev() {
                                    let qty = self.depth.bid_qty_at_tick(t);
                                    if qty > 0.0 {
                                        let exec_qty = qty.min(order.leaves_qty);
                                        self.fill::<true>(order, timestamp, false, t, exec_qty)?;
                                    }
                                    if order.status == Status::Filled {
                                        return Ok(());
                                    }
                                }

                                // The sell order cannot remain in the bid book, as it cannot affect the
                                // market depth during backtesting based on market-data replay. So, even
                                // though it simulates partial fill, if the order size is not small enough,
                                // it introduces unreality.
                                let (price_tick, leaves_qty) = (order.price_tick, order.leaves_qty);
                                self.fill::<true>(order, timestamp, false, price_tick, leaves_qty)
                            }
                            _ => {
                                unreachable!();
                            }
                        }
                    } else {
                        match order.time_in_force {
                            TimeInForce::GTC | TimeInForce::GTX => {
                                // Initializes the order's queue position.
                                self.queue_model.new_order(order, &self.depth);
                                order.status = Status::New;
                                // The exchange accepts this order.
                                self.sell_orders
                                    .entry(order.price_tick)
                                    .or_default()
                                    .insert(order.order_id);

                                order.exch_timestamp = timestamp;
                                self.orders
                                    .borrow_mut()
                                    .insert(order.order_id, order.clone());
                                Ok(())
                            }
                            TimeInForce::FOK | TimeInForce::IOC => {
                                order.status = Status::Expired;
                                order.exch_timestamp = timestamp;
                                Ok(())
                            }
                            TimeInForce::Unsupported => Err(BacktestError::InvalidOrderRequest),
                        }
                    }
                }
                OrdType::Market => {
                    // todo: set the proper lower bound.
                    for t in ((self.depth.best_bid_tick() - 100)..=self.depth.best_bid_tick()).rev()
                    {
                        let qty = self.depth.bid_qty_at_tick(t);
                        if qty > 0.0 {
                            let exec_qty = qty.min(order.leaves_qty);
                            self.fill::<true>(order, timestamp, false, t, exec_qty)?;
                        }
                        if order.status == Status::Filled {
                            return Ok(());
                        }
                    }
                    order.status = Status::Expired;
                    order.exch_timestamp = timestamp;
                    Ok(())
                }
                OrdType::Unsupported => Err(BacktestError::InvalidOrderRequest),
            }
        }
    }

    fn ack_cancel(&mut self, order: &mut Order, timestamp: i64) -> Result<(), BacktestError> {
        let exch_order = {
            let mut order_borrowed = self.orders.borrow_mut();
            order_borrowed.remove(&order.order_id)
        };

        if exch_order.is_none() {
            order.req = Status::Rejected;
            order.exch_timestamp = timestamp;
            return Ok(());
        }

        let exch_order = exch_order.unwrap();
        let _ = std::mem::replace(order, exch_order);

        // Deletes the order.
        if order.side == Side::Buy {
            self.buy_orders
                .get_mut(&order.price_tick)
                .unwrap()
                .remove(&order.order_id);
        } else {
            self.sell_orders
                .get_mut(&order.price_tick)
                .unwrap()
                .remove(&order.order_id);
        }
        order.status = Status::Canceled;
        order.exch_timestamp = timestamp;
        Ok(())
    }

    fn ack_modify<const RESET_QUEUE_POS: bool>(
        &mut self,
        order: &mut Order,
        timestamp: i64,
    ) -> Result<(), BacktestError> {
        let (prev_order_price_tick, prev_leaves_qty) = {
            let order_borrowed = self.orders.borrow();
            let exch_order = order_borrowed.get(&order.order_id);

            // The order can be already deleted due to fill or expiration.
            if exch_order.is_none() {
                order.req = Status::Rejected;
                order.exch_timestamp = timestamp;
                return Ok(());
            }

            let exch_order = exch_order.unwrap();
            (exch_order.price_tick, exch_order.leaves_qty)
        };

        // The initialization of the order queue position may not occur when the modified quantity
        // is smaller than the previous quantity, depending on the exchanges. It may need to
        // implement exchange-specific specialization.
        if RESET_QUEUE_POS
            || prev_order_price_tick != order.price_tick
            || order.qty > prev_leaves_qty
        {
            let mut cancel_order = order.clone();
            self.ack_cancel(&mut cancel_order, timestamp)?;
            self.ack_new(order, timestamp)?;
            // todo: Status::Replaced or Status::New?
        } else {
            let mut order_borrowed = self.orders.borrow_mut();
            let exch_order = order_borrowed.get_mut(&order.order_id);
            let exch_order = exch_order.unwrap();

            exch_order.qty = order.qty;
            exch_order.leaves_qty = order.qty;
            exch_order.exch_timestamp = timestamp;
            // todo: Status::Replaced or Status::New?
            exch_order.status = Status::New;
            order.leaves_qty = order.qty;
            order.exch_timestamp = timestamp;
            order.status = Status::New;
        }
        Ok(())
    }
}

impl<AT, LM, QM, MD, FM> Processor for PartialFillExchange<AT, LM, QM, MD, FM>
where
    AT: AssetType,
    LM: LatencyModel,
    QM: QueueModel<MD>,
    MD: MarketDepth + L2MarketDepth,
    FM: FeeModel,
{
    fn event_seen_timestamp(&self, event: &Event) -> Option<i64> {
        event.is(EXCH_EVENT).then_some(event.exch_ts)
    }

    fn process(&mut self, event: &Event) -> Result<(), BacktestError> {
        if event.is(EXCH_BID_DEPTH_CLEAR_EVENT) {
            self.depth.clear_depth(Side::Buy, event.px);
            self.consumed_bid.clear();
        } else if event.is(EXCH_ASK_DEPTH_CLEAR_EVENT) {
            self.depth.clear_depth(Side::Sell, event.px);
            self.consumed_ask.clear();
        } else if event.is(EXCH_DEPTH_CLEAR_EVENT) {
            self.depth.clear_depth(Side::None, 0.0);
            self.consumed_bid.clear();
            self.consumed_ask.clear();
        } else if event.is(EXCH_BID_DEPTH_EVENT) || event.is(EXCH_BID_DEPTH_SNAPSHOT_EVENT) {
            let (price_tick, prev_best_bid_tick, best_bid_tick, prev_qty, new_qty, timestamp) =
                self.depth
                    .update_bid_depth(event.px, event.qty, event.exch_ts);
            own_fills_after_level_update(
                &mut self.consumed_bid,
                self.own_fill_depletion_persists,
                price_tick,
                self.depth.bid_qty_at_tick(price_tick),
            );
            self.on_bid_qty_chg(price_tick, prev_qty, new_qty);
            if best_bid_tick > prev_best_bid_tick {
                self.on_best_bid_update(prev_best_bid_tick, best_bid_tick, timestamp)?;
            }
        } else if event.is(EXCH_ASK_DEPTH_EVENT) || event.is(EXCH_ASK_DEPTH_SNAPSHOT_EVENT) {
            let (price_tick, prev_best_ask_tick, best_ask_tick, prev_qty, new_qty, timestamp) =
                self.depth
                    .update_ask_depth(event.px, event.qty, event.exch_ts);
            own_fills_after_level_update(
                &mut self.consumed_ask,
                self.own_fill_depletion_persists,
                price_tick,
                self.depth.ask_qty_at_tick(price_tick),
            );
            self.on_ask_qty_chg(price_tick, prev_qty, new_qty);
            if best_ask_tick < prev_best_ask_tick {
                self.on_best_ask_update(prev_best_ask_tick, best_ask_tick, timestamp)?;
            }
        } else if event.is(EXCH_BUY_TRADE_EVENT) {
            let price_tick = (event.px / self.depth.tick_size()).round() as i64;
            let qty = event.qty;
            {
                let orders = self.orders.clone();
                let mut orders_borrowed = orders.borrow_mut();
                if self.depth.best_bid_tick() == INVALID_MIN
                    || (orders_borrowed.len() as i64) < price_tick - self.depth.best_bid_tick()
                {
                    for (_, order) in orders_borrowed.iter_mut() {
                        if order.side == Side::Sell {
                            self.check_if_sell_filled(order, price_tick, qty, event.exch_ts)?;
                        }
                    }
                } else {
                    for t in (self.depth.best_bid_tick() + 1)..=price_tick {
                        if let Some(order_ids) = self.sell_orders.get(&t) {
                            for order_id in order_ids.clone().iter() {
                                let order = orders_borrowed.get_mut(order_id).unwrap();
                                self.check_if_sell_filled(order, price_tick, qty, event.exch_ts)?;
                            }
                        }
                    }
                }
            }
            self.remove_filled_orders();
        } else if event.is(EXCH_SELL_TRADE_EVENT) {
            let price_tick = (event.px / self.depth.tick_size()).round() as i64;
            let qty = event.qty;
            {
                let orders = self.orders.clone();
                let mut orders_borrowed = orders.borrow_mut();
                if self.depth.best_ask_tick() == INVALID_MAX
                    || (orders_borrowed.len() as i64) < self.depth.best_ask_tick() - price_tick
                {
                    for (_, order) in orders_borrowed.iter_mut() {
                        if order.side == Side::Buy {
                            self.check_if_buy_filled(order, price_tick, qty, event.exch_ts)?;
                        }
                    }
                } else {
                    for t in (price_tick..self.depth.best_ask_tick()).rev() {
                        if let Some(order_ids) = self.buy_orders.get(&t) {
                            for order_id in order_ids.clone().iter() {
                                let order = orders_borrowed.get_mut(order_id).unwrap();
                                self.check_if_buy_filled(order, price_tick, qty, event.exch_ts)?;
                            }
                        }
                    }
                }
            }
            self.remove_filled_orders();
        }

        Ok(())
    }

    fn process_recv_order(
        &mut self,
        timestamp: i64,
        _wait_resp_order_id: Option<OrderId>,
    ) -> Result<bool, BacktestError> {
        while let Some(mut order) = self.order_e2l.receive(timestamp) {
            // Processes a new order.
            if order.req == Status::New {
                if order.time_in_force == TimeInForce::IOC {
                    if let Some(batches) = &self.ioc_batches {
                        let batch = batches.receive(order.order_id);
                        self.active_ioc_batch.is_none_or(|active| batch >= active)
                            .then_some(())
                            .expect("IOC order bus must deliver submitted batches contiguously in submission order");
                        if self.active_ioc_batch != Some(batch) {
                            self.consumed_bid.clear();
                            self.consumed_ask.clear();
                            self.active_ioc_batch = Some(batch);
                        }
                    }
                }
                order.req = Status::None;
                self.ack_new(&mut order, timestamp)?;
            }
            // Processes a cancel order.
            else if order.req == Status::Canceled {
                order.req = Status::None;
                self.ack_cancel(&mut order, timestamp)?;
            }
            // Processes a modify order.
            else if order.req == Status::Replaced {
                order.req = Status::None;
                self.ack_modify::<false>(&mut order, timestamp)?;
            } else {
                return Err(BacktestError::InvalidOrderRequest);
            }
            // Makes the response.
            self.order_e2l.respond(order);
        }
        Ok(false)
    }

    fn earliest_recv_order_timestamp(&self) -> i64 {
        self.order_e2l
            .earliest_recv_order_timestamp()
            .unwrap_or(i64::MAX)
    }

    fn earliest_send_order_timestamp(&self) -> i64 {
        self.order_e2l
            .earliest_send_order_timestamp()
            .unwrap_or(i64::MAX)
    }
}

#[cfg(test)]
mod own_fill_depletion_tests {
    use std::collections::HashMap;

    use super::{own_fill_available, own_fills_after_level_update};

    #[test]
    fn submitted_batches_have_distinct_identity_and_release_registration_at_arrival() {
        let batches = super::IocBatchRegistry::default();
        let first = batches.begin_batch();
        let second = batches.begin_batch();
        assert!(second > first);
        batches.register_order(42, first).unwrap();
        assert!(batches.register_order(42, second).is_err());
        assert_eq!(batches.receive(42), first);
        assert!(batches.0.borrow().orders.is_empty());
        batches.register_order(42, second).unwrap();
        assert_eq!(batches.receive(42), second);
    }

    #[test]
    #[should_panic(expected = "every accepted IOC must have its submitted batch registered")]
    fn unregistered_ioc_stops_at_exchange_arrival() {
        super::IocBatchRegistry::default().receive(42);
    }

    fn taken(qty: f64) -> HashMap<i64, f64> {
        HashMap::from([(834_124, qty)])
    }

    #[test]
    fn a_feed_update_makes_the_level_whole_by_default() {
        let mut consumed = taken(0.03);
        own_fills_after_level_update(&mut consumed, false, 834_124, 0.045);
        assert!(consumed.is_empty());
    }

    #[test]
    fn persisting_keeps_what_we_took_across_a_feed_update() {
        let mut consumed = taken(0.03);
        own_fills_after_level_update(&mut consumed, true, 834_124, 0.045);
        assert_eq!(consumed[&834_124], 0.03);
        // Another level's update leaves it alone.
        own_fills_after_level_update(&mut consumed, true, 834_125, 0.0);
        assert_eq!(consumed[&834_124], 0.03);
    }

    #[test]
    fn persisting_caps_what_we_took_at_the_displayed_quantity() {
        let mut consumed = taken(0.03);
        own_fills_after_level_update(&mut consumed, true, 834_124, 0.01);
        assert_eq!(consumed[&834_124], 0.01);
        // Quantity added afterwards is available beyond what we took.
        own_fills_after_level_update(&mut consumed, true, 834_124, 0.05);
        assert_eq!(consumed[&834_124], 0.01);
    }

    #[test]
    fn a_fully_taken_level_offers_nothing_whatever_rounding_the_fills_left() {
        // 3.06427 displayed; fills of 0.94783 and of the rest add up to one unit in the last place short of it.
        let (displayed, first) = (3.06427_f64, 0.94783_f64);
        let taken = first + (displayed - first);
        assert!(
            displayed - taken > 0.0,
            "the case must leave a positive remainder"
        );
        assert_eq!(own_fill_available(displayed, taken, true, 0.00001), 0.0);
        // A real remainder is available in full, and the default rule is the plain difference.
        assert_eq!(own_fill_available(1.0, 0.6, true, 0.00001), 1.0 - 0.6);
        assert_eq!(
            own_fill_available(displayed, taken, false, 0.00001),
            displayed - taken
        );
    }

    #[test]
    fn persisting_starts_fresh_once_the_feed_empties_the_level() {
        let mut consumed = taken(0.03);
        own_fills_after_level_update(&mut consumed, true, 834_124, 0.0);
        assert!(consumed.is_empty());
    }
}
