type pass = Loop_program.t -> Loop_program.t

(* Not alphabetical: each pass exposes what the next one uses. Unit loops
   first, turning their variables into constants for folding; folding and the
   range-aware simplification before the guards pass, whose proofs read the
   simplified bounds; CSE while accesses are still per-axis coordinates it can
   compare; hoisting before collapsing, since a loop emptied of its invariant
   statements may become perfectly nested; collapsing last, as it flattens
   the coordinates everything before it reads. *)
let passes : pass list =
  [
    Loop_opt_unit_loops.run;
    Loop_opt_fold.run;
    Loop_opt_simplify.run;
    Loop_opt_guards.run;
    Loop_opt_cse.run;
    Loop_opt_hoist.run;
    Loop_opt_collapse.run;
  ]

let run ?(passes = passes) program =
  List.fold_left (fun program pass -> pass program) program passes
