# =============================================================================
# Notification Service — Full LLD Implementation
# Consolidates all code from the HTML study guide into one runnable file.
#
# Features:
#   - Push / Email / SMS channels (Strategy pattern)
#   - Priority-based delivery via min-heap (HIGH=1 pops first)
#   - Retry on failure with exponential backoff + jitter
#   - Channel registry (Registry pattern)
#   - Dead-letter tracking for exhausted retries
# =============================================================================

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List
import heapq, time, random

random.seed(42)  # reproducible output for smoke test


# ── Enums ─────────────────────────────────────────────────────────────────────

class ChannelType(Enum):
    PUSH  = "push"
    EMAIL = "email"
    SMS   = "sms"


class Priority(IntEnum):
    # IntEnum so Priority.HIGH < Priority.LOW works (1 < 3)
    # Min-heap pops the smallest value first → HIGH priority goes first
    HIGH   = 1
    MEDIUM = 2
    LOW    = 3


# ── Notification ──────────────────────────────────────────────────────────────

@dataclass
class Notification:
    id:           str
    recipient:    str
    message:      str
    priority:     Priority
    channel_type: ChannelType

    def __lt__(self, other: "Notification") -> bool:
        # heapq needs this to compare two Notification objects when
        # priorities are equal — without it, Python crashes at runtime
        return self.priority < other.priority


# ── Abstract Channel (the interface the interviewer cared about) ──────────────

class NotificationChannel(ABC):
    """
    Contract that every delivery channel must fulfill.
    NotificationService depends on THIS, never on PushChannel directly.
    That's the D in SOLID (Dependency Inversion).
    """

    @property
    @abstractmethod
    def channel_type(self) -> ChannelType:
        pass

    @abstractmethod
    def send(self, notification: Notification) -> bool:
        """Return True on success, False on failure."""
        pass


# ── Concrete Channels ─────────────────────────────────────────────────────────

class PushChannel(NotificationChannel):
    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.PUSH

    def send(self, n: Notification) -> bool:
        # In prod: call FCM/APNs API; return False on HTTPError/timeout
        success = random.random() > 0.3  # 70% success rate
        status  = "OK  " if success else "FAIL"
        print(f"    [PUSH /{status}] → {n.recipient}: {n.message}")
        return success


class EmailChannel(NotificationChannel):
    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.EMAIL

    def send(self, n: Notification) -> bool:
        # In prod: call SMTP / SendGrid API
        success = random.random() > 0.2  # 80% success rate
        status  = "OK  " if success else "FAIL"
        print(f"    [EMAIL/{status}] → {n.recipient}: {n.message}")
        return success


class SMSChannel(NotificationChannel):
    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.SMS

    def send(self, n: Notification) -> bool:
        # In prod: call Twilio API
        success = random.random() > 0.1  # 90% success rate
        status  = "OK  " if success else "FAIL"
        print(f"    [SMS  /{status}] → {n.recipient}: {n.message}")
        return success


# ── Retry Policy (Exponential Backoff + Jitter) ───────────────────────────────

class RetryPolicy:
    """
    Single Responsibility: owns ALL retry math, nothing else.
    The service dispatches; this class decides delays.

    Delay formula: base_delay * 2^attempt
      attempt 0 → base * 1   (e.g. 1.0s)
      attempt 1 → base * 2   (e.g. 2.0s)
      attempt 2 → base * 4   (e.g. 4.0s)

    Jitter: adds ±10% random offset to spread retrying clients
    and avoid the thundering herd problem.
    """

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay  = base_delay

    def get_delay(self, attempt: int) -> float:
        base   = self.base_delay * (2 ** attempt)
        jitter = random.uniform(0, base * 0.1)  # ±10% jitter
        return base + jitter


# ── Notification Service (the orchestrator) ───────────────────────────────────

class NotificationService:
    """
    Responsibilities:
      - Maintain a priority queue (min-heap)
      - Route each notification to the right channel
      - Retry on failure using the injected RetryPolicy
      - Track permanently failed notifications (dead-letter)
    """

    def __init__(self, retry_policy: RetryPolicy = None):
        self._queue:        List               = []   # min-heap
        self._channels:     Dict[ChannelType, NotificationChannel] = {}
        self._retry_policy: RetryPolicy        = retry_policy or RetryPolicy()
        self._dead_letter:  List[Notification] = []   # exhausted retries land here

    def register_channel(self, channel: NotificationChannel) -> None:
        """Registry pattern — map ChannelType to its handler."""
        self._channels[channel.channel_type] = channel
        print(f"  ✦ Registered channel: {channel.channel_type.value.upper()}")

    def enqueue(self, notification: Notification) -> None:
        """Push to min-heap. HIGH priority (=1) pops out first."""
        heapq.heappush(self._queue, notification)

    def process_all(self) -> None:
        """Drain the queue in priority order, retrying failures."""
        print(f"\n{'─'*55}")
        print(f"  Processing {len(self._queue)} notification(s)...")
        print(f"{'─'*55}")
        while self._queue:
            n = heapq.heappop(self._queue)
            print(f"\n  [{n.priority.name}] '{n.id}' → {n.channel_type.value.upper()}")
            self._dispatch_with_retry(n)

        if self._dead_letter:
            print(f"\n  ⚠  Dead-letter queue ({len(self._dead_letter)} failed):")
            for n in self._dead_letter:
                print(f"     - {n.id} ({n.recipient})")

    def _dispatch_with_retry(self, n: Notification) -> None:
        """
        Template Method skeleton:
          get channel → loop → call send() → sleep (backoff) → repeat
        The actual send() behaviour is defined by each channel subclass.
        """
        channel = self._channels.get(n.channel_type)
        if not channel:
            print(f"    [ERROR] No channel registered for {n.channel_type}")
            self._dead_letter.append(n)
            return

        policy = self._retry_policy
        for attempt in range(policy.max_retries + 1):
            if channel.send(n):
                print(f"    ✓ Delivered on attempt {attempt + 1}")
                return

            if attempt < policy.max_retries:
                delay = policy.get_delay(attempt)
                print(f"    ✗ Failed. Retry {attempt + 2}/{policy.max_retries + 1} in {delay:.2f}s...")
                time.sleep(delay)

        print(f"    ✗ All {policy.max_retries + 1} attempts exhausted → dead-letter")
        self._dead_letter.append(n)


# ── Smoke Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Notification Service LLD — Smoke Test")
    print("=" * 55)

    # --- Setup ---
    # base_delay=0.05 so retries are fast during testing
    # Change to 1.0 for realistic production behaviour
    policy  = RetryPolicy(max_retries=3, base_delay=0.05)
    service = NotificationService(retry_policy=policy)

    print("\nRegistering channels:")
    service.register_channel(PushChannel())
    service.register_channel(EmailChannel())
    service.register_channel(SMSChannel())

    # --- Enqueue out-of-order; service must reorder by priority ---
    print("\nEnqueueing notifications (out of priority order):")

    service.enqueue(Notification(
        "n1", "bob@x.com", "Weekly newsletter",
        Priority.LOW, ChannelType.EMAIL
    ))
    print("  + n1 [LOW]    Email  → bob")

    service.enqueue(Notification(
        "n2", "alice@x.com", "Your OTP is 9821",
        Priority.HIGH, ChannelType.SMS
    ))
    print("  + n2 [HIGH]   SMS    → alice")

    service.enqueue(Notification(
        "n3", "carol@x.com", "Order shipped!",
        Priority.MEDIUM, ChannelType.PUSH
    ))
    print("  + n3 [MEDIUM] Push   → carol")

    service.enqueue(Notification(
        "n4", "dave@x.com", "Password reset link",
        Priority.HIGH, ChannelType.EMAIL
    ))
    print("  + n4 [HIGH]   Email  → dave")

    service.enqueue(Notification(
        "n5", "eve@x.com", "Flash sale — 50% off!",
        Priority.LOW, ChannelType.PUSH
    ))
    print("  + n5 [LOW]    Push   → eve")

    # --- Process — expected order: n2/n4 (HIGH) → n3 (MEDIUM) → n1/n5 (LOW) ---
    service.process_all()

    print("\n" + "=" * 55)
    print("  Smoke test complete.")
    print("=" * 55)