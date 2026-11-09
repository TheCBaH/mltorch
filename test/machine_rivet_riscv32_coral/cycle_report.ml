(* Where a report of simulated cycles goes. This suite runs on the RTL
   testbench, whose cycle count is the core's, so it prints them. The MPACT
   variant of the suite (test/machine_rivet_riscv32_coral_mpact) replaces this
   module with one that discards them: MPACT has no timing model, and what it
   checks is the result. *)

let pr fmt = Fmt.pr fmt
