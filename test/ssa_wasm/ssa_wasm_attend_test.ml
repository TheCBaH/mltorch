open Ssa_bridge
open Ssa_ir
module Fx = Ssa_ir_test.Ssa_fixtures
module Lf = Loop_ir_test.Loop_fixtures
module E = Ssa_ir_test.Ssa_expand_attend_test
module A = Attention_block

(* A binary32 attention, expanded and narrowed, through generated WebAssembly: the output
   is [Attention_block.eval]'s, bit for bit. *)

let case ~precision ~mask ~tq ~tk ~d ~bk =
  let dsc = E.desc ~precision ~mask ~tq ~tk ~d ~bk in
  let buffers =
    [
      E.f32 0 ~h:tq ~w:d Ssa_buffer.Input;
      E.f32 1 ~h:tk ~w:d Ssa_buffer.Input;
      E.f32 2 ~h:tk ~w:d Ssa_buffer.Input;
      E.f32 3 ~h:tq ~w:d Ssa_buffer.Output;
    ]
    @ if mask then [ E.f32 4 ~h:tq ~w:tk Ssa_buffer.Input ] else []
  in
  let p =
    Ssa_precision.to_f32 ~exact:true
      (Ssa_expand_attend.program (E.program ~buffers ~mask dsc))
  in
  let inputs =
    [
      (0, Array.init (tq * d) (fun _ -> E.value ()));
      (1, Array.init (tk * d) (fun _ -> E.value ()));
      (2, Array.init (tk * d) (fun _ -> E.value ()));
      (4, Array.init (tq * tk) (fun _ -> E.mask_value ()));
    ]
  in
  let dims = [ (0, (tq, d)); (1, (tk, d)); (2, (tk, d)); (4, (tq, tk)) ] in
  let cells id = List.assoc id inputs in
  let want = Array.make (tq * d) 0. in
  let at a o = a.(Int64.to_int o) in
  A.eval dsc
    ~q:(at (cells 0))
    ~k:(at (cells 1))
    ~v:(at (cells 2))
    ~mask:(at (cells 4))
    ~write:(fun o x -> want.(Int64.to_int o) <- x);
  let lowered =
    match Ssa_wasm.lower p with
    | Ok k -> k
    | Error e -> Fmt.failwith "%a" Ssa_wasm.pp_error e
  in
  let loop =
    Lf.program
      ~buffers:(List.map Loop_of_ssa.loop_buffer (Ssa_wasm.arguments p))
      []
  in
  let bind id =
    List.find_map
      (fun (n, cs) ->
        if Tensor_id.equal id (Lf.tid n) then
          let h, w = List.assoc n dims in
          let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h ~w ~c:1 in
          Some
            (Lf.f32_tensor shape (fun c -> cs.((Vec6.offset shape c :> int))))
        else None)
      inputs
  in
  match Err.payload (Loop_wasm_exec.exec_module ~lowered loop ~bind) with
  | Error e -> Fmt.failwith "%a" Loop_wasm_exec.pp_error e
  | Ok outputs ->
      let got =
        match Tensor_id.Map.find_opt (Lf.tid 3) outputs with
        | Some t ->
            Array.init (tq * d) (fun i ->
                Tensor.read t
                  (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:(i / d) ~w:(i mod d) ~c:0))
        | None -> [||]
      in
      Fmt.pr "%s %dx%d d%d bk%d%s: Wasm equals the reference: %b@."
        (A.Precision.name precision)
        tq tk d bk
        (if mask then " masked" else "")
        (Array.length got = tq * d
        && Array.for_all2 Core.Float_bits.equal_portable got want)

let%expect_test "an expanded attention through Wasm equals the reference" =
  List.iter
    (fun precision ->
      List.iter
        (fun (tq, tk, d, bk, mask) -> case ~precision ~mask ~tq ~tk ~d ~bk)
        [ (5, 7, 4, 4, false); (3, 8, 6, 4, true); (9, 29, 8, 8, true) ])
    [ A.Precision.F32_ordered; A.Precision.F32_fused ];
  [%expect
    {|
    f32_ordered 5x7 d4 bk4: Wasm equals the reference: true
    f32_ordered 3x8 d6 bk4 masked: Wasm equals the reference: true
    f32_ordered 9x29 d8 bk8 masked: Wasm equals the reference: true
    f32_fused 5x7 d4 bk4: Wasm equals the reference: true
    f32_fused 3x8 d6 bk4 masked: Wasm equals the reference: true
    f32_fused 9x29 d8 bk8 masked: Wasm equals the reference: true |}]
