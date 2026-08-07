# Review: detailed Model Explorer implementation plan, revision 26

Reviewed document:
`~/.claude/plans/create-detailed-implementation-plan-dazzling-squirrel.md`
(revision 26, 5,682 lines; SHA-256
`956f4ceae3fead6bac477abbaabe140e8f866d46748614e2d388f7e2a8093e0d`).

Review basis:

- `.ai/model_explorer_plan.md`, `.ai/js_backends_design.md`, `.ai/testing_strategy.md`,
  and `.ai/jsont_patterns.md`;
- the current source at `3f0c10ad0c8950683ed996b5a4664a47d78d380c`;
- the repository's module, result-boundary, non-vacuity, and 32-bit-jsoo rules;
- every finding in the revision-25 review.

## Verdict

Revision 26 resolves both revision-25 state-coherence findings. The candidate-to-loaded handoff is
now an exact operation postcondition with a genuinely falsifying mutation, and ticket identity is
separated from registry-owned state with an explicit linked-machine invariant. It also correctly
returns the mount check to JavaScript, treats stale generation as defence in depth, and expands
prepare-failure assertions over the retained scalar/session state.

The native stages and 0A spike remain ready, and §6 should proceed to code rather than another prose
round. Stage 2 cannot freeze exactly as written, however: the new mount-precheck failure fixture asks
`abort` to preserve a visible DOM state that the fixture itself has already destroyed. Two smaller
test specifications also remain inconsistent with the new resolution: stale generation is still
listed as a public coordinator sequence, and prepare-failure atomicity does not preserve an
unrelated live ticket.

## Blocking finding

### P1. Mount-precheck failure cannot leave the old element visible by calling `abort` alone

Plan references: X4 resolution lines 76-83; installation sequence lines 3669-3688; mount-precheck
ownership lines 3697-3704; JSDOM case lines 4080-4084.

The revised ownership is correct: only the coordinator can inspect the DOM. The specified failure
path is not coherent. The precheck fails when the mount is disconnected or `current` is no longer
its child. The test creates exactly those conditions by detaching the mount or replacing its child,
then requires:

- `abort(token)`;
- no `commit` and no coordinator `replaceChild`;
- the old element still shown;
- unchanged bridge state.

`abort` changes only the OCaml installation/ticket state. It cannot reconnect a detached mount or
put `current` back after the fixture replaced it. Consequently the old element is already not
reliably visible before `abort` runs, and the stated postcondition is impossible without another DOM
operation. Worse, returning to an idle bridge after plain abort leaves the authoritative old loaded
session beside a missing or unrelated visible node—the inconsistency the transaction is designed to
contain.

Choose and test an actual recovery policy in the implementation:

- attempt the captured, bounded, postcondition-verified restoration of `current`, then call `abort`
  only after restoration succeeds; or
- treat a failed precheck as coordinator corruption and enter the existing disposal/recreation/reload
  path, with bridge reset only after the DOM reaches its specified empty state.

If the mount's privacy is intended to make failure impossible in production, the check may remain a
defensive assertion, but its adversarial fixture still needs a satisfiable containment result. A
plain `abort` plus “old element still shown” is not one.

## Major verification findings

### P2. `Stale_generation` is still specified as a reachable public sequence

Plan references: X3 resolution lines 69-74; defence-in-depth classification lines 3445-3450;
sequence-test list lines 3975-3980; Stage-2 test gate line 5224.

X3 correctly says global installation serialization means no public transition can create a stale
staged candidate and relabels the comparison as an internal-corruption guard. The old coordinator
test remains in two places as “commit after a newer root has committed, rejected by
`Stale_generation`,” including the Stage-2 list of public sequence tests.

That prerequisite cannot be constructed through the published API: while the first candidate is
`Staged`, another `prepare*` is rejected with `Install_in_progress`; reset clears the installation;
and committing the staged candidate consumes it. Move the stale check to an internal
`Me_js_session` guard test using a test-only corrupted state (or remove the impossible public
sequence). Keep cancellation/invalidation during `modelGraphProcessed` as the separate reachable
coordinator case.

### P2. Prepare-failure “whole state” coverage omits unrelated tickets

Plan references: X5 resolution lines 84-89; linked-machine relation lines 3460-3466;
linked-ticket tests lines 4060-4065; prepare-failure tests lines 4073-4079.

The new failure test preserves `generation`, `loaded`, and `last_settlement` and proves the consumed
ticket is removed. It does not prove that only that ticket is removed. `tickets` is retained runtime
state too, and a faulty rollback implemented as `tickets <- []` would satisfy the listed assertions:
there is no target ticket, the next same-kind build succeeds, and all three retained scalar/session
fields are unchanged.

Run each injected prepare failure with a second ticket of the other kind live. Assert that the target
consumed entry is removed while the unrelated entry, including its state, remains unchanged and can
complete normally. The linked-machine tests already use this fixture for settlement and abort; add
prepare's internal rollback to the same exact-removal matrix. A red/restore that clears the full
registry should make the case fail.

## Confirmed revision-26 corrections

- The five loaded/installation rules are now operation postconditions stated as equalities.
  Successful prepare preserves loaded state and generation; commit installs exactly
  `Candidate.proposed`; rollback restores exactly `Previous.t`; finalize preserves the committed
  loaded value; reset clears loaded and installation together.
- The commit test retains the staged proposed value and deliberately substitutes another valid
  `Loaded.t`, satisfying the repository's non-vacuity requirement.
- `Ticket.Handle.t` now carries immutable `{ id; kind }`, `Ticket.t` is the sole registry entry that
  carries state, and `Candidate.t` stores only the handle, base, and proposed session. The public token
  is derived rather than duplicated.
- `Installation.handle` makes the one-consumed-ticket relation expressible uniformly for Preparing,
  Staged, Committed, and Mount-inconsistent states; settlement and cancellation are checked against a
  second live ticket.
- The first-open representation, declaration order, shared settlement result, exported
  `maxRestoreSteps`, poisoned state, bounded DOM restoration, and first-open tests remain coherent.
- Mount inspection is correctly assigned to the JavaScript coordinator immediately before the
  synchronous bridge commit; the bridge API itself is no longer claimed to inspect DOM state.
- Prepare failure now preserves generation, loaded state, and the prior settlement marker, and the
  successful-prepare sequence no longer falsely says all retained state is untouched.
- Remote JSON, local and gated real-browser `.pt2`, CORS/object-storage policy, JSDOM and Playwright
  coverage, native CLI parity, transformation flow, and deep graph/detail navigation remain grounded
  and unchanged.

## Recommendation

Proceed with native implementation and the 0A spike, then implement §6 in OCaml and TypeScript as
planned. Before Stage 2 freezes, give pre-commit mount corruption a real DOM recovery/disposal path,
move `Stale_generation` to an explicitly internal guard test, and prove prepare failure preserves the
unrelated ticket. Record the compiled state representation and the chosen DOM recovery policy in
`.ai/model_explorer_design.md`; no further revision of this 5,682-line working plan is needed.
