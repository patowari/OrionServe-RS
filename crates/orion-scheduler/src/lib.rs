//! Continuous batching inference scheduler.
//!
//! The scheduler decides, every engine step, which sequences run and with how
//! many tokens. It is the component that turns a stream of independent requests
//! into batches large enough to keep a GPU busy, without letting any one
//! request monopolize it.
//!
//! See `docs/scheduler.md` for the policy rationale and complexity analysis.
//!
//! # Testability
//!
//! Nothing here touches a GPU or a model. The scheduler is generic over
//! [`KvCacheManagerLike`](orion_core::KvCacheManagerLike), so its policy can be
//! driven against the real cache manager or against
//! [`testing::FakeCache`], which forces allocation failures on demand.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod queue;
pub mod scheduler;
pub mod testing;

pub use queue::{RunningQueue, WaitingQueue};
pub use scheduler::{ScheduledSequence, Scheduler, SchedulerOutput, SchedulerStats};
pub use testing::FakeCache;
