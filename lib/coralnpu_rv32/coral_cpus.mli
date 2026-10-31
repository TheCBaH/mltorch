(** A host-wide pool of CPUs for the Coral simulators, shared by every process
    that runs one: a run holds as many of the pool's slots as its simulator
    keeps busy ({!Coral_profile.t.threads}) for as long as it runs, so a test
    runner or a build that starts runs in parallel cannot oversubscribe the host
    (on eight CPUs, two 4-thread subsystem simulators or eight single-threaded
    cores at once).

    The pool is lock files in [/tmp/mltorch-coral-cpus-<uid>] (not under
    [TMPDIR], which dune gives each build its own), one per CPU: the online
    CPUs, or [MLTORCH_CORAL_CPUS]. A slot is a POSIX record lock, so a process
    that dies releases its slots. Slots are taken all or none, waiting while too
    few are free; a request past the pool takes the whole pool. *)

val with_cpus : int -> (unit -> 'a) -> 'a
(** [with_cpus n f] runs [f] holding [n] slots (at least one). *)

val pool : unit -> int
(** The pool's size. *)
