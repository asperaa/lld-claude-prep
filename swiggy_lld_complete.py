"""
Swiggy LLD — Complete Code (Consolidated)
==========================================
All classes from the guide in one place, in dependency order:
  1. Enums & Exceptions
  2. Value Objects (Address, OrderItem)
  3. Domain Classes (User, MenuItem, Payment, DeliveryAgent, Order, Restaurant)
  4. Interfaces (ABC) — Observer & Strategy
  5. Concrete Observers
  6. Concrete Strategies
  7. SwiggySystem (orchestrator)
  8. Quick smoke-test
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from math import sqrt
from typing import Dict, List, Optional, Tuple
from uuid import uuid4


# ─────────────────────────────────────────────
# 1. ENUMS & EXCEPTIONS
# ─────────────────────────────────────────────

class OrderStatus(Enum):
    PLACED    = auto()
    ACCEPTED  = auto()
    PREPARING = auto()
    READY     = auto()
    PICKED_UP = auto()
    DELIVERED = auto()
    CANCELLED = auto()


class PaymentStatus(Enum):
    PENDING  = auto()
    PAID     = auto()
    REFUNDED = auto()


class AgentStatus(Enum):
    AVAILABLE   = auto()
    ON_DELIVERY = auto()
    OFFLINE     = auto()


# Custom exceptions — always raise, never return False on invalid state
class UserNotFoundError(Exception):        pass
class RestaurantNotFoundError(Exception):  pass
class RestaurantClosedError(Exception):    pass
class InsufficientStockError(Exception):   pass
class InvalidTransitionError(Exception):   pass
class NoAgentAvailableError(Exception):    pass
class OrderNotFoundError(Exception):       pass
class CancellationNotAllowedError(Exception): pass


# ─────────────────────────────────────────────
# 2. VALUE OBJECTS
# ─────────────────────────────────────────────
@dataclass
class Address:
    """Pure data — no behavior, no mutable state."""

    street: str
    city: str
    pincode: str
    latitude: float
    longitude: float

@dataclass(frozen=True)
class OrderItem:
    """
    Immutable price snapshot.
    This guarantees historical order accuracy regardless of future price changes.
    """
    item_id: str
    name: str
    quantity: int
    price_at_order: float

    @property
    def subtotal(self) -> float:
        """Derived, never stored — always correct, can never go stale."""
        return self.quantity * self.price_at_order


# ─────────────────────────────────────────────
# 3. DOMAIN CLASSES
# ─────────────────────────────────────────────

class Payment:
    """
    Separate from Order even though it's 1:1 today.
    Reason: Order changes when food lifecycle changes.
             Payment changes when money lifecycle changes.
    Different teams, different rates of change → separate classes.
    """
    def __init__(self, order_id: str, amount: float):
        self.payment_id = str(uuid4())
        self.order_id   = order_id
        self.amount     = amount
        self.status     = PaymentStatus.PENDING

    def mark_paid(self):
        self.status = PaymentStatus.PAID

    def mark_refunded(self):
        self.status = PaymentStatus.REFUNDED


class MenuItem:
    """
    Full class (not a dict) because inventory rules must be encapsulated here.
    If MenuItem were a plain dict, callers would scatter inventory checks
    everywhere — some would forget to validate, others would allow negatives.
    """
    def __init__(self, item_id: str, name: str, price: float,
                 inventory_count: int, is_available: bool = True):
        self.item_id         = item_id
        self.name            = name
        self.price           = price
        self.inventory_count = inventory_count   # int, not float — no 2.5 biryanis
        self.is_available    = is_available

    def is_in_stock(self, qty: int) -> bool:
        """
        Checks BOTH is_available (manual pause flag) AND inventory_count.
        Callers get both checks for free — they cannot forget one.
        """
        return self.is_available and self.inventory_count >= qty

    def deduct_stock(self, qty: int):
        """
        Re-validates even though place_order() already checked.
        Design by contract: methods own their own preconditions.
        Protects against any future code path that skips the external check.
        """
        if not self.is_in_stock(qty):
            raise InsufficientStockError(
                f"{self.name}: requested {qty}, available {self.inventory_count}"
            )
        self.inventory_count -= qty

    def restore_stock(self, qty: int):
        """Called on cancellation. Single point of change for restore logic."""
        self.inventory_count += qty


class User:
    def __init__(self, user_id: str, name: str, address: Address):
        self.user_id  = user_id
        self.name     = name
        self.address  = address
        self.order_history: List[str] = []   # list of order_ids

    def add_order_to_history(self, order_id: str):
        self.order_history.append(order_id)


class DeliveryAgent:
    """
    Three-state availability (not a boolean) because OFFLINE and ON_DELIVERY
    are meaningfully different: OFFLINE won't come back, ON_DELIVERY will.
    """
    def __init__(self, agent_id: str, name: str, location: Tuple[float, float]):
        self.agent_id        = agent_id
        self.name            = name
        self.location        = location          # (lat, lng)
        self.status          = AgentStatus.AVAILABLE
        self.current_order_id: Optional[str] = None
        self.rating          = 5.0
        self.deliveries_today = 0

    def is_available(self) -> bool:
        """
        Encapsulates the definition of 'available'.
        Tomorrow this could also check fatigue limit or zone — callers
        never need to change, they always call is_available().
        """
        return self.status == AgentStatus.AVAILABLE

    def assign_order(self, order_id: str):
        """
        Updates BOTH fields atomically in one method.
        No window where status=ON_DELIVERY but current_order_id=None.
        """
        self.current_order_id = order_id
        self.status           = AgentStatus.ON_DELIVERY

    def complete_delivery(self):
        """
        Frees the agent for the next assignment.
        Single place to add any completion logic (e.g. log delivery duration).
        """
        self.deliveries_today += 1
        self.current_order_id = None
        self.status           = AgentStatus.AVAILABLE


class Order:
    """
    The most complex class. Owns its own state machine — transition rules
    are data (VALID_TRANSITIONS dict), not scattered if/else.
    """

    # State machine as data — one dict to rule all transitions.
    # DELIVERED and CANCELLED are absent (terminal states — no exits).
    VALID_TRANSITIONS: Dict[OrderStatus, List[OrderStatus]] = {
        OrderStatus.PLACED:    [OrderStatus.ACCEPTED, OrderStatus.CANCELLED],
        OrderStatus.ACCEPTED:  [OrderStatus.PREPARING, OrderStatus.CANCELLED],
        OrderStatus.PREPARING: [OrderStatus.READY],
        OrderStatus.READY:     [OrderStatus.PICKED_UP],
        OrderStatus.PICKED_UP: [OrderStatus.DELIVERED],
    }

    def __init__(self, order_id: str, user_id: str, restaurant_id: str,
                 items: List[OrderItem], observers: Optional[List["OrderObserver"]] = None):
        self.order_id          = order_id        # UUID, not int — unguessable
        self.user_id           = user_id
        self.restaurant_id     = restaurant_id
        self.items             = items
        self.status            = OrderStatus.PLACED
        self.created_at        = datetime.now()  # needed for SLA, auto-cancel, analytics
        self.delivery_agent_id: Optional[str] = None
        self.payment:          Optional[Payment] = None
        self._observers        = observers or []

    # ── State machine ──────────────────────────────────────────

    def update_status(self, new_status: OrderStatus):
        """
        Raises InvalidTransitionError on illegal transitions.
        Never returns False — silent failure hides bugs.
        Fires all registered observers after the transition.
        """
        allowed = self.VALID_TRANSITIONS.get(self.status, [])
        if new_status not in allowed:
            raise InvalidTransitionError(
                f"Cannot transition from {self.status} to {new_status}"
            )
        self.status = new_status
        for obs in self._observers:
            obs.on_status_change(self, new_status)

    def can_cancel(self) -> bool:
        """
        Derived directly from VALID_TRANSITIONS — no separate hard-coded list.
        This is the single source of truth for cancellability.
        Adding a new state with CANCELLED in its exits? can_cancel() auto-updates.
        """
        return OrderStatus.CANCELLED in self.VALID_TRANSITIONS.get(self.status, [])

    # ── Business logic ─────────────────────────────────────────

    def calculate_total(self) -> float:
        """
        Computed from OrderItems — never stored separately.
        'Derive, don't store' — impossible to go stale.
        """
        return sum(item.subtotal for item in self.items)

    def add_observer(self, obs: "OrderObserver"):
        self._observers.append(obs)


class Restaurant:
    def __init__(self, restaurant_id: str, name: str, city: str,
                 cuisine: str, location: Tuple[float, float]):
        self.restaurant_id = restaurant_id
        self.name          = name
        self.city          = city      # string, not a class — no behavior needed
        self.cuisine       = cuisine   # same
        self.location      = location  # (lat, lng)
        self.is_open       = True
        self.menu: List[MenuItem] = []

    def add_menu_item(self, item: MenuItem):
        self.menu.append(item)

    def open(self):
        self.is_open = True

    def close(self):
        self.is_open = False


# ─────────────────────────────────────────────
# 4. INTERFACES (ABCs)
# ─────────────────────────────────────────────

class OrderObserver(ABC):
    """
    Observer pattern interface.
    Order knows nothing about push/SMS/analytics — it just calls this.
    Adding a new notification channel = new class, zero edits to Order.
    """
    @abstractmethod
    def on_status_change(self, order: Order, new_status: OrderStatus): ...


class AgentStrategy(ABC):
    """
    Strategy pattern interface.
    SwiggySystem doesn't care which algorithm is wired in — it just calls find_agent().
    Swap algorithms at construction time or at runtime.
    """
    @abstractmethod
    def find_agent(self, agents: List[DeliveryAgent], order: Order) -> DeliveryAgent: ...


# ─────────────────────────────────────────────
# 5. CONCRETE OBSERVERS
# ─────────────────────────────────────────────

class PushNotificationObserver(OrderObserver):
    def on_status_change(self, order: Order, new_status: OrderStatus):
        messages = {
            OrderStatus.ACCEPTED:  "Your order has been accepted!",
            OrderStatus.PREPARING: "The restaurant is preparing your food.",
            OrderStatus.READY:     "Food is ready — agent on the way!",
            OrderStatus.PICKED_UP: "Your order has been picked up.",
            OrderStatus.DELIVERED: "Your order has been delivered. Enjoy!",
            OrderStatus.CANCELLED: "Your order was cancelled.",
        }
        msg = messages.get(new_status)
        if msg:
            print(f"[PUSH → user {order.user_id}] {msg}")


class PaymentObserver(OrderObserver):
    """Captures payment on DELIVERED, refunds on CANCELLED."""
    def on_status_change(self, order: Order, new_status: OrderStatus):
        if order.payment is None:
            return
        if new_status == OrderStatus.DELIVERED:
            order.payment.mark_paid()
            print(f"[PAYMENT] Order {order.order_id[:8]}… marked PAID ₹{order.calculate_total():.2f}")
        elif new_status == OrderStatus.CANCELLED:
            order.payment.mark_refunded()
            print(f"[PAYMENT] Order {order.order_id[:8]}… REFUNDED ₹{order.calculate_total():.2f}")


class AnalyticsObserver(OrderObserver):
    def on_status_change(self, order: Order, new_status: OrderStatus):
        print(f"[ANALYTICS] Order {order.order_id[:8]}… → {new_status.name}")


# ─────────────────────────────────────────────
# 6. CONCRETE STRATEGIES
# ─────────────────────────────────────────────

def _euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class NearestAgentStrategy(AgentStrategy):
    """
    Good tier: nearest available agent to the restaurant.
    Correct for most interviews. Explain and defend this one.
    """
    def find_agent(self, agents: List[DeliveryAgent], order: Order) -> DeliveryAgent:
        available = [a for a in agents if a.is_available()]
        if not available:
            raise NoAgentAvailableError("No available delivery agents.")
        restaurant = None
        # get restaurant location from order — in a full system you'd look it up
        # here we pass it via order.restaurant_location (set by SwiggySystem)
        rest_loc = getattr(order, '_restaurant_location', (0.0, 0.0))
        return min(available, key=lambda a: _euclidean_distance(a.location, rest_loc))


class WeightedScoringStrategy(AgentStrategy):
    """
    Great tier: weighted score across distance, load, and rating.
    Weights are config — tunable without touching algorithm code.
    """
    W_DISTANCE = 0.6
    W_LOAD     = 0.25
    W_RATING   = 0.15

    def find_agent(self, agents: List[DeliveryAgent], order: Order) -> DeliveryAgent:
        available = [a for a in agents if a.is_available()]
        if not available:
            raise NoAgentAvailableError()

        rest_loc = getattr(order, '_restaurant_location', (0.0, 0.0))

        def score(agent: DeliveryAgent) -> float:
            dist_score   = 1 / (1 + _euclidean_distance(agent.location, rest_loc))
            load_score   = 1 / (1 + agent.deliveries_today)
            rating_score = agent.rating / 5.0
            return (self.W_DISTANCE * dist_score +
                    self.W_LOAD     * load_score +
                    self.W_RATING   * rating_score)

        return max(available, key=score)


# ─────────────────────────────────────────────
# 7. SWIGGY SYSTEM — ORCHESTRATOR
# ─────────────────────────────────────────────

class SwiggySystem:
    """
    Facade / single entry point for all operations.
    Owns all registries. Coordinates cross-entity operations.
    Delegates business rules to the right entity — never does the work itself.

    Dependencies are injected (Dependency Inversion Principle):
      - strategy: swap agent algorithms without touching this class
      - observers: swap/add notification channels without touching this class
    """

    def __init__(self, strategy: Optional[AgentStrategy] = None,
                 default_observers: Optional[List[OrderObserver]] = None):
        self.restaurants: Dict[str, Restaurant]     = {}
        self.users:       Dict[str, User]           = {}
        self.agents:      Dict[str, DeliveryAgent]  = {}
        self.orders:      Dict[str, Order]          = {}
        self.strategy    = strategy or NearestAgentStrategy()
        self._default_observers = default_observers or [
            PushNotificationObserver(),
            PaymentObserver(),
            AnalyticsObserver(),
        ]

    # ── Registration ───────────────────────────────────────────

    def register_user(self, user: User):
        self.users[user.user_id] = user

    def register_restaurant(self, restaurant: Restaurant):
        self.restaurants[restaurant.restaurant_id] = restaurant

    def register_agent(self, agent: DeliveryAgent):
        self.agents[agent.agent_id] = agent

    # ── Core operation: place_order ─────────────────────────────
    #
    #   Two-phase design:
    #     Phase 1 — read-only validation (nothing mutated)
    #     Phase 2 — mutation (only reached if Phase 1 passed fully)
    #
    #   Guard ordering within Phase 1:
    #     1. structural checks (existence)  — O(1), cheapest
    #     2. operational checks (state)     — O(1)
    #     3. per-item stock                 — O(N), most expensive
    #
    #   This order gives users the clearest possible error message and
    #   ensures Phase 2 never needs a rollback.

    def place_order(self, user_id: str, restaurant_id: str,
                    items: List[Tuple[MenuItem, int]]) -> Order:
        # ── Phase 1: Read-only validation ──────────────────────

        # Guard 1: structural — does the user exist?
        user = self.users.get(user_id)
        if not user:
            raise UserNotFoundError(f"User '{user_id}' not found.")

        # Guard 2: structural — does the restaurant exist?
        restaurant = self.restaurants.get(restaurant_id)
        if not restaurant:
            raise RestaurantNotFoundError(f"Restaurant '{restaurant_id}' not found.")

        # Guard 3: operational — is the restaurant open?
        # (Checked AFTER existence — can't call .is_open on None)
        if not restaurant.is_open:
            raise RestaurantClosedError(f"'{restaurant.name}' is currently closed.")

        # Guard 4: operational — check ALL items before touching ANY stock
        for item, qty in items:
            if not item.is_in_stock(qty):
                raise InsufficientStockError(
                    f"{item.name}: requested {qty}, available {item.inventory_count}"
                )

        # ── Phase 2: Mutation (Phase 1 passed fully — no rollback needed) ──

        # Deduct stock first — if this fails (race condition), no Order is created
        for item, qty in items:
            item.deduct_stock(qty)

        # Snapshot prices into OrderItems — immutable records at this exact moment
        order_items = [
            OrderItem(
                item_id=item.item_id,
                name=item.name,
                quantity=qty,
                price_at_order=item.price,   # frozen at order time
            )
            for item, qty in items
        ]

        # Create Order — status starts at PLACED, not ACCEPTED
        # (restaurant hasn't responded yet)
        order = Order(
            order_id=str(uuid4()),
            user_id=user_id,
            restaurant_id=restaurant_id,
            items=order_items,
            observers=list(self._default_observers),
        )

        # Attach restaurant location (used by agent assignment strategy)
        order._restaurant_location = restaurant.location

        # Create Payment — only after Order exists (PENDING until DELIVERED)
        order.payment = Payment(
            order_id=order.order_id,
            amount=order.calculate_total(),
        )

        # Register and return — must be stored before returning so
        # any subsequent operation (cancel, status update) can find it
        self.orders[order.order_id] = order
        user.add_order_to_history(order.order_id)
        return order

    # ── cancel_order ────────────────────────────────────────────

    def cancel_order(self, order_id: str) -> bool:
        order = self.orders.get(order_id)
        if not order:
            raise OrderNotFoundError(order_id)

        # can_cancel() lives on Order (Tell, Don't Ask) —
        # the rule "cancellable before PREPARING" belongs to Order, not here.
        if not order.can_cancel():
            raise CancellationNotAllowedError(
                f"Order in status {order.status.name} cannot be cancelled."
            )

        # Restore inventory for each item
        restaurant = self.restaurants.get(order.restaurant_id)
        if restaurant:
            item_map = {m.item_id: m for m in restaurant.menu}
            for oi in order.items:
                menu_item = item_map.get(oi.item_id)
                if menu_item:
                    menu_item.restore_stock(oi.quantity)

        # Transition triggers observers: PaymentObserver will mark REFUNDED,
        # PushObserver will notify the user.
        order.update_status(OrderStatus.CANCELLED)
        return True

    # ── update_order_status ─────────────────────────────────────

    def update_order_status(self, order_id: str, new_status: OrderStatus,
                            actor_id: Optional[str] = None) -> bool:
        order = self.orders.get(order_id)
        if not order:
            raise OrderNotFoundError(order_id)

        order.update_status(new_status)

        # Auto-assign agent when food is READY
        # (assign at READY, not ACCEPTED — minimises agent idle time at restaurants)
        if new_status == OrderStatus.READY:
            self._assign_delivery_agent(order_id)

        # Free the agent when DELIVERED
        if new_status == OrderStatus.DELIVERED:
            if order.delivery_agent_id:
                agent = self.agents.get(order.delivery_agent_id)
                if agent:
                    agent.complete_delivery()

        return True

    # ── assign_delivery_agent ───────────────────────────────────

    def _assign_delivery_agent(self, order_id: str):
        order = self.orders.get(order_id)
        if not order:
            raise OrderNotFoundError(order_id)

        # Strategy pattern — SwiggySystem never knows which algorithm is running
        agent = self.strategy.find_agent(list(self.agents.values()), order)

        # Update BOTH sides of the relationship (O(1) lookup in either direction)
        agent.assign_order(order_id)
        order.delivery_agent_id = agent.agent_id
        print(f"[DISPATCH] Agent '{agent.name}' assigned to order {order_id[:8]}…")


# ─────────────────────────────────────────────
# 8. SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Swiggy LLD — Smoke Test")
    print("=" * 60)

    # ── Setup ──────────────────────────────────────────────────
    system = SwiggySystem(strategy=NearestAgentStrategy())

    user = User("u1", "Aditya", Address("12 MG Road", "Bengaluru", "560001", 12.97, 77.59))
    system.register_user(user)

    restaurant = Restaurant("r1", "Biryani House", "Bengaluru", "North Indian", (12.96, 77.60))
    biryani = MenuItem("m1", "Chicken Biryani", 220.0, inventory_count=5)
    naan    = MenuItem("m2", "Butter Naan",      40.0, inventory_count=10)
    restaurant.add_menu_item(biryani)
    restaurant.add_menu_item(naan)
    system.register_restaurant(restaurant)

    agent = DeliveryAgent("a1", "Ravi Kumar", location=(12.965, 77.605))
    system.register_agent(agent)

    # ── Place order ─────────────────────────────────────────────
    print("\n--- Placing order ---")
    order = system.place_order("u1", "r1", [(biryani, 2), (naan, 1)])
    print(f"Order placed: {order.order_id[:8]}… | Total: ₹{order.calculate_total():.2f}")
    print(f"Biryani stock after order: {biryani.inventory_count}")  # 3

    # ── Walk through status transitions ─────────────────────────
    print("\n--- Restaurant accepts ---")
    system.update_order_status(order.order_id, OrderStatus.ACCEPTED)

    print("\n--- Cooking starts ---")
    system.update_order_status(order.order_id, OrderStatus.PREPARING)

    print("\n--- Food ready (agent auto-assigned) ---")
    system.update_order_status(order.order_id, OrderStatus.READY)
    print(f"Agent assigned: {order.delivery_agent_id}")

    print("\n--- Agent picks up ---")
    system.update_order_status(order.order_id, OrderStatus.PICKED_UP)

    print("\n--- Delivered! ---")
    system.update_order_status(order.order_id, OrderStatus.DELIVERED)
    print(f"Payment status: {order.payment.status.name}")
    print(f"Agent status after delivery: {agent.status.name}")

    # ── Test cancellation ────────────────────────────────────────
    print("\n--- Placing a second order (to test cancel) ---")
    biryani2 = MenuItem("m3", "Veg Biryani", 180.0, inventory_count=3)
    restaurant.add_menu_item(biryani2)
    order2 = system.place_order("u1", "r1", [(biryani2, 1)])
    print(f"Order2 placed | Status: {order2.status.name}")

    print("\n--- Cancelling order2 ---")
    system.cancel_order(order2.order_id)
    print(f"Order2 status: {order2.status.name}")
    print(f"Veg Biryani stock restored: {biryani2.inventory_count}")  # back to 3

    # ── Invalid transition guard ─────────────────────────────────
    print("\n--- Trying to cancel DELIVERED order (should fail) ---")
    try:
        system.cancel_order(order.order_id)
    except CancellationNotAllowedError as e:
        print(f"Correctly blocked: {e}")

    print("\n✓ All checks passed.")
