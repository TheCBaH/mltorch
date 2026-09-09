(* Shared fixtures for the Const-SSA test modules in this directory (split
   from the former const_ssa_test.ml). Not a test module itself. *)

open Graph_ir

let arena = Ground_expr.Arena.create ()
let t_ = Tensor_id.of_int
let value id = Const_ssa.Value_id.of_tensor_id (t_ id)
let f32 = Payload.Fmt Payload.F32
let sig_ id shape = Tensor_sig.create ~id:(t_ id) ~name:"" ~shape ~fmt:f32 ()

let ramp shape =
  Tensor.materialize shape (fun c ->
      float_of_int
        ((Dim.to_int (Vec6.get c Axis.H) * 10) + Dim.to_int (Vec6.get c Axis.W)))

let swap_hw =
  [
    (Axis.N, Axis.N);
    (Axis.T, Axis.T);
    (Axis.D, Axis.D);
    (Axis.H, Axis.W);
    (Axis.W, Axis.H);
    (Axis.C, Axis.C);
  ]

let pp_result pp = function
  | Ok () -> Format.printf "ok@."
  | Error e -> Format.printf "%a@." pp (Err.Error.kind e)
