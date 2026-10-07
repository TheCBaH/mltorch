(* The interpreters' models of [Mir_math]'s helpers: the host's binary64
   libm, the declared binding, on the argument's exact bits. *)

open Machine_ir

let host = function
  | Mir_math.Fn.Cos -> Stdlib.cos
  | Mir_math.Fn.Exp -> Stdlib.exp
  | Mir_math.Fn.Log -> Stdlib.log
  | Mir_math.Fn.Sin -> Stdlib.sin

let model fn =
  let f = host fn in
  {
    Mir_helper_model.name = Mir_math.Fn.name fn;
    version = Mir_math.version;
    run =
      (fun _ -> function
        | [ Mir_datum.Bits b ] ->
            Mir_helper_model.Returns
              [ Mir_datum.f64 (f (Int64.float_of_bits b)) ]
        | _ ->
            invalid_arg
              "Mir_math_model: an argument the descriptor's signature rejects");
  }

let all = List.map model Mir_math.Fn.all
