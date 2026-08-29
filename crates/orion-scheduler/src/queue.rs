//! The waiting and running queues.
//!
//! Both are thin wrappers over standard collections. They exist as named types
//! because the scheduler's fairness properties are properties of *these*
//! orderings, and burying them in a bare `VecDeque` in the scheduler makes
//! those properties impossible to test in isolation.

use std::collections::VecDeque;

use orion_core::{Sequence, SequenceId};

/// FIFO queue of sequences awaiting their first schedule, or awaiting
/// rescheduling after preemption.
///
/// # Fairness
///
/// Strict FIFO. A preempted sequence goes to the *front*, not the back: it has
/// already waited once and already consumed prefill compute, so sending it to
/// the back would let a steady arrival stream starve it indefinitely while
/// repeatedly throwing away its work. This is the queue's single most important
/// property, and the `preempted_sequences_regain_priority` test pins it.
#[derive(Debug, Default)]
pub struct WaitingQueue {
    inner: VecDeque<Sequence>,
}

impl WaitingQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Enqueues a newly arrived sequence at the back.
    pub fn push_back(&mut self, seq: Sequence) {
        self.inner.push_back(seq);
    }

    /// Re-enqueues a preempted sequence at the front, preserving its priority.
    pub fn push_front(&mut self, seq: Sequence) {
        self.inner.push_front(seq);
    }

    /// Removes and returns the next sequence to consider.
    pub fn pop_front(&mut self) -> Option<Sequence> {
        self.inner.pop_front()
    }

    /// Inspects the next sequence without removing it.
    ///
    /// The scheduler peeks before popping because admission depends on whether
    /// the sequence *fits* in the remaining budget; popping first would mean
    /// pushing it back on a miss, which would disturb the ordering.
    pub fn peek(&self) -> Option<&Sequence> {
        self.inner.front()
    }

    /// Removes a specific sequence, e.g. on client cancellation.
    ///
    /// Linear in queue length. Cancellation is rare relative to scheduling, so
    /// this is not on any hot path.
    pub fn remove(&mut self, id: SequenceId) -> Option<Sequence> {
        let pos = self.inner.iter().position(|s| s.id() == id)?;
        self.inner.remove(pos)
    }

    /// Removes and returns every sequence for which `pred` holds, preserving
    /// the order of those that remain.
    pub fn drain_where<F>(&mut self, mut pred: F) -> Vec<Sequence>
    where
        F: FnMut(&Sequence) -> bool,
    {
        let mut removed = Vec::new();
        let mut kept = VecDeque::with_capacity(self.inner.len());
        for seq in self.inner.drain(..) {
            if pred(&seq) {
                removed.push(seq);
            } else {
                kept.push_back(seq);
            }
        }
        self.inner = kept;
        removed
    }

    pub fn iter(&self) -> impl Iterator<Item = &Sequence> {
        self.inner.iter()
    }
}

/// The set of sequences currently holding KV blocks and a batch slot.
///
/// Insertion order is preserved so that decode scheduling is round-robin
/// stable: a sequence does not lose or gain priority merely by having been
/// stepped.
#[derive(Debug, Default)]
pub struct RunningQueue {
    inner: Vec<Sequence>,
}

impl RunningQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn push(&mut self, seq: Sequence) {
        self.inner.push(seq);
    }

    pub fn get(&self, id: SequenceId) -> Option<&Sequence> {
        self.inner.iter().find(|s| s.id() == id)
    }

    pub fn get_mut(&mut self, id: SequenceId) -> Option<&mut Sequence> {
        self.inner.iter_mut().find(|s| s.id() == id)
    }

    pub fn remove(&mut self, id: SequenceId) -> Option<Sequence> {
        let pos = self.inner.iter().position(|s| s.id() == id)?;
        Some(self.inner.remove(pos))
    }

    /// Removes the most recently admitted sequence.
    ///
    /// This is the preemption victim policy: last-in, first-out. The newest
    /// sequence has generated the fewest tokens, so evicting it discards the
    /// least work, and it cannot starve because it goes to the front of the
    /// waiting queue.
    pub fn pop_newest(&mut self) -> Option<Sequence> {
        self.inner.pop()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Sequence> {
        self.inner.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Sequence> {
        self.inner.iter_mut()
    }

    /// Removes and returns every finished sequence.
    pub fn take_finished(&mut self) -> Vec<Sequence> {
        let mut finished = Vec::new();
        let mut i = 0;
        while i < self.inner.len() {
            if self.inner[i].is_finished() {
                finished.push(self.inner.remove(i));
            } else {
                i += 1;
            }
        }
        finished
    }

    /// Total tokens currently held across all running sequences. Used for
    /// cache-pressure reporting.
    pub fn total_tokens(&self) -> usize {
        self.inner.iter().map(|s| s.total_len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orion_core::{SamplingParams, SequenceState};

    fn seq(prompt_len: usize) -> Sequence {
        Sequence::new(vec![1; prompt_len], SamplingParams::default())
    }

    #[test]
    fn waiting_queue_is_fifo_for_new_arrivals() {
        let mut q = WaitingQueue::new();
        let a = seq(1);
        let b = seq(2);
        let (ida, idb) = (a.id(), b.id());
        q.push_back(a);
        q.push_back(b);

        assert_eq!(q.pop_front().unwrap().id(), ida);
        assert_eq!(q.pop_front().unwrap().id(), idb);
        assert!(q.pop_front().is_none());
    }

    #[test]
    fn preempted_sequences_regain_priority() {
        // The starvation-regression test: a preempted sequence must not be sent
        // to the back of a queue that new arrivals keep refilling.
        let mut q = WaitingQueue::new();
        for _ in 0..5 {
            q.push_back(seq(1));
        }
        let preempted = seq(9);
        let id = preempted.id();
        q.push_front(preempted);

        assert_eq!(
            q.pop_front().unwrap().id(),
            id,
            "preempted sequence must be scheduled before newer arrivals"
        );
    }

    #[test]
    fn repeated_preemption_cannot_starve_a_sequence() {
        // Simulates the pathological loop: a sequence is preempted, new work
        // arrives, and it is preempted again. It must still run before any of
        // the newer arrivals.
        let mut q = WaitingQueue::new();
        let victim = seq(100);
        let id = victim.id();
        q.push_back(victim);

        for round in 0..10 {
            let s = q.pop_front().unwrap();
            assert_eq!(s.id(), id, "starved on round {round}");
            q.push_back(seq(1)); // a new arrival appears
            q.push_front(s); // the victim is preempted again
        }
        assert_eq!(q.pop_front().unwrap().id(), id);
    }

    #[test]
    fn peek_does_not_consume() {
        let mut q = WaitingQueue::new();
        let s = seq(3);
        let id = s.id();
        q.push_back(s);

        assert_eq!(q.peek().unwrap().id(), id);
        assert_eq!(q.len(), 1);
        assert_eq!(q.pop_front().unwrap().id(), id);
    }

    #[test]
    fn removal_targets_one_sequence_and_preserves_order() {
        let mut q = WaitingQueue::new();
        let a = seq(1);
        let b = seq(2);
        let c = seq(3);
        let (ida, idb, idc) = (a.id(), b.id(), c.id());
        q.push_back(a);
        q.push_back(b);
        q.push_back(c);

        assert_eq!(q.remove(idb).unwrap().id(), idb);
        assert_eq!(q.len(), 2);
        assert_eq!(q.pop_front().unwrap().id(), ida);
        assert_eq!(q.pop_front().unwrap().id(), idc);
    }

    #[test]
    fn removing_an_absent_sequence_returns_none() {
        let mut q = WaitingQueue::new();
        q.push_back(seq(1));
        assert!(q.remove(SequenceId::from_raw(9999)).is_none());
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn drain_where_partitions_and_keeps_order() {
        let mut q = WaitingQueue::new();
        for len in [1, 10, 2, 20, 3] {
            q.push_back(seq(len));
        }
        let long = q.drain_where(|s| s.prompt_len() >= 10);
        assert_eq!(long.len(), 2);
        let remaining: Vec<_> = q.iter().map(|s| s.prompt_len()).collect();
        assert_eq!(remaining, vec![1, 2, 3]);
    }

    #[test]
    fn running_queue_evicts_the_newest_sequence_first() {
        let mut q = RunningQueue::new();
        let a = seq(1);
        let b = seq(2);
        let (ida, idb) = (a.id(), b.id());
        q.push(a);
        q.push(b);

        assert_eq!(
            q.pop_newest().unwrap().id(),
            idb,
            "the newest sequence has the least work to lose"
        );
        assert_eq!(q.pop_newest().unwrap().id(), ida);
        assert!(q.pop_newest().is_none());
    }

    #[test]
    fn take_finished_removes_only_completed_sequences() {
        use orion_core::FinishReason;
        let mut q = RunningQueue::new();
        for _ in 0..4 {
            q.push(seq(2));
        }
        let ids: Vec<_> = q.iter().map(|s| s.id()).collect();
        q.get_mut(ids[1]).unwrap().finish(FinishReason::Stop);
        q.get_mut(ids[3]).unwrap().finish(FinishReason::Length);

        let finished = q.take_finished();
        assert_eq!(finished.len(), 2);
        assert_eq!(q.len(), 2);
        let left: Vec<_> = q.iter().map(|s| s.id()).collect();
        assert_eq!(left, vec![ids[0], ids[2]]);
    }

    #[test]
    fn total_tokens_sums_prompt_and_output() {
        let mut q = RunningQueue::new();
        let mut a = seq(5);
        a.push_token(1);
        a.push_token(2);
        q.push(a);
        q.push(seq(3));
        assert_eq!(q.total_tokens(), 7 + 3);
    }

    #[test]
    fn running_queue_lookup_by_id() {
        let mut q = RunningQueue::new();
        let s = seq(4);
        let id = s.id();
        q.push(s);

        assert!(q.get(id).is_some());
        assert_eq!(q.get(id).unwrap().state(), SequenceState::Waiting);
        assert!(q.get(SequenceId::from_raw(4242)).is_none());
        assert_eq!(q.remove(id).unwrap().id(), id);
        assert!(q.is_empty());
    }
}
