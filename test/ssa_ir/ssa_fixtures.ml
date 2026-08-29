open Ssa_ir

(* Small hand-built programs shared by the SSA tests. Everything is tiny and
   independently computable: a test that needs a model belongs to the
   source-lowering suites, not here. *)

let buf = Ssa_id.Buffer.of_int
let extents ~h ~w = Expr.Coord.make ~n:1L ~t:1L ~d:1L ~h ~w ~c:1L

let buffer id ~h ~w format role =
  { Ssa_buffer.id = buf id; extents = extents ~h ~w; format; role }

(* A coordinate that varies along H and W only. *)
let at b ~h ~w =
  let zero = Ssa_builder.index b 0L in
  Ssa_builder.Coord (Expr.Coord.make ~n:zero ~t:zero ~d:zero ~h ~w ~c:zero)

let floats a = Ssa_memory.Floats a

let memory bindings =
  List.fold_left
    (fun m (id, cells) -> Ssa_id.Buffer.Map.add (buf id) cells m)
    Ssa_id.Buffer.Map.empty bindings

let build ~buffers f =
  Err.or_raise ~pp_error:Ssa_verify.pp_error (Ssa_builder.program ~buffers f)

let run ?counters p ~memory =
  Err.or_raise ~pp_error:Ssa_interp.pp_error
    (Ssa_interp.run ?counters p ~memory)

let run_result ?counters p ~memory =
  Err.payload (Ssa_interp.run ?counters p ~memory)

(* C[m,n] = round_f32 (sum over k of A[m,k] * B[k,n]), A:M*K, B:K*N, as the
   reference lowering of a matmul: an ordered sum seeded at +0, one reduction
   mark per term, an F32 store. *)
let matmul ~m ~k ~n =
  let a =
    buffer 0 ~h:(Int64.of_int m) ~w:(Int64.of_int k) Ssa_format.F32
      Ssa_buffer.Input
  in
  let b =
    buffer 1 ~h:(Int64.of_int k) ~w:(Int64.of_int n) Ssa_format.F32
      Ssa_buffer.Input
  in
  let c =
    buffer 2 ~h:(Int64.of_int m) ~w:(Int64.of_int n) Ssa_format.F32
      Ssa_buffer.Output
  in
  build ~buffers:[ a; b; c ] (fun bld ->
      let zero = Ssa_builder.index bld 0L in
      let rows = Ssa_builder.index bld (Int64.of_int m) in
      let cols = Ssa_builder.index bld (Int64.of_int n) in
      let depth = Ssa_builder.index bld (Int64.of_int k) in
      let Ssa_builder.Nil =
        Ssa_builder.for_ bld ~lo:zero ~hi:rows ~init:Ssa_builder.Nil
          (fun bld i Ssa_builder.Nil ->
            let Ssa_builder.Nil =
              Ssa_builder.for_ bld ~lo:zero ~hi:cols ~init:Ssa_builder.Nil
                (fun bld j Ssa_builder.Nil ->
                  let seed = Ssa_builder.f64 bld 0. in
                  let sum =
                    Ssa_builder.ordered_sum bld ~lo:zero ~hi:depth ~seed
                      (fun bld p ->
                        Ssa_builder.mark bld Ssa_mark.Reduction;
                        let x =
                          Ssa_builder.load_f64 bld (buf 0)
                            ~decode:Ssa_op.Decode.F32_to_f64 (at bld ~h:i ~w:p)
                        in
                        let y =
                          Ssa_builder.load_f64 bld (buf 1)
                            ~decode:Ssa_op.Decode.F32_to_f64 (at bld ~h:p ~w:j)
                        in
                        Ssa_builder.f64_binary bld Expr.Value.Mul x y)
                  in
                  Ssa_builder.store_f64 bld (buf 2)
                    ~encode:Ssa_op.Encode.F32_round (at bld ~h:i ~w:j) sum;
                  Ssa_builder.Nil)
            in
            Ssa_builder.Nil)
      in
      ())

(* The same sum written by hand, for the independent expectation. *)
let matmul_expected ~m ~k ~n a b =
  Array.init (m * n) (fun at ->
      let i = at / n and j = at mod n in
      let acc = ref 0. in
      for p = 0 to k - 1 do
        acc := !acc +. (a.((i * k) + p) *. b.((p * n) + j))
      done;
      Ssa_const.round_f32 !acc)

let output_floats memory id =
  match Ssa_memory.find memory (buf id) with
  | Some (Ssa_memory.Floats a) -> a
  | Some (Ssa_memory.Int64s _ | Ssa_memory.Ints _) | None ->
      invalid_arg "output_floats"
