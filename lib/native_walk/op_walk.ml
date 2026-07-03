(* ATen specialization of the shared walk runner: the subject is an
   [Aten_spec.Op_spec.t] and verification runs it through the ATen-vs-native
   harness ([Aten_spec_run.run]). The generic loop lives in [Walk_core.Walk]. *)

let run m ~ppf ~pcg ~steps =
  let verify ppf spec = Aten_spec_run.run ~ppf spec in
  Walk_core.Walk.run m ~verify ~ppf ~pcg ~steps
