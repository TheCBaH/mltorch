module Cases = Coral_estimate_cases
module CE = Coral_estimate.Coral_census
module M = Machine_model.Mir_model

(* every invocation's census, in schedule order, as the Coral route plans the
   case *)
let censuses ?(ram = fun _ -> []) (c : Cases.t) =
  let b =
    Err.or_raise ~pp_error:Loop_ir.Loop_bundle.pp_error
      (match c.Cases.layer with
      | None -> Loop_ir.Loop_bundle.build c.Cases.graph
      | Some layer ->
          Loop_ir.Loop_bundle.build
            ~granted:(fun id -> Option.is_some (layer id))
            c.Cases.graph)
  in
  (* the calibration's own record of where invocation [i]'s buffers were
     placed, else the case's home *)
  let region i (b : Ssa_ir.Ssa_id.Buffer.t) =
    match c.Cases.home with
    | Cases.Home.Ram -> CE.Region.Ram
    | Cases.Home.Default ->
        if List.mem (b :> int) (ram i) then CE.Region.Ram else CE.Region.Dtcm
  in
  let out = ref [] in
  let route =
    let module R = Machine_model.Mir_model_route in
    R.Route.Custom
      {
        R.Custom.name = "VME final SSA, generic execution";
        plan =
          (fun p ->
            Err.payload
              (Ssa_ir.Ssa_contract_plan.program
                 ~capabilities:
                   {
                     Contraction.Select.Capabilities.matrix = true;
                     vector = true;
                   }
                 p)
            |> Result.map_error (Fmt.str "%a" Ssa_ir.Ssa_contract_plan.pp_error));
        exec = (fun ~sites:_ g -> Ok (R.generic g));
      }
  in
  match
    M.prepare ~route ~contractions:c.Cases.contractions
      ~bf16_pairs:c.Cases.bf16_pairs ?attention:c.Cases.attention
      ?layer:c.Cases.layer ~pipeline:(Cases.pipeline c)
      ~on_program:(fun _ p _ ->
        out := CE.program ~region:(region (List.length !out)) p :: !out)
      b
  with
  | Error rs -> Error (Fmt.str "%a" Fmt.(list ~sep:(any "; ") M.Refusal.pp) rs)
  | Ok _ -> Ok (List.rev !out)

let%expect_test "every case plans" =
  List.iter
    (fun (c : Cases.t) ->
      match censuses c with
      | Error e -> Fmt.pr "%s: refused: %s@." c.Cases.name e
      | Ok cs ->
          let unknown =
            List.fold_left (fun n (x : CE.t) -> n + x.CE.unknown) 0 cs
          in
          Fmt.pr "%s: %d invocation(s)%s@." c.Cases.name (List.length cs)
            (if unknown > 0 then Fmt.str ", %d unknown" unknown else ""))
    Cases.all;
  [%expect
    {|
    cos 16x16: 1 invocation(s)
    cos 16x16, scalar: 1 invocation(s)
    cos 32x32: 1 invocation(s)
    cos 32x32, scalar: 1 invocation(s)
    exp 16x16: 1 invocation(s)
    exp 16x16, scalar: 1 invocation(s)
    exp 32x32: 1 invocation(s)
    exp 32x32, scalar: 1 invocation(s)
    log 16x16: 1 invocation(s)
    log 16x16, scalar: 1 invocation(s)
    log 32x32: 1 invocation(s)
    log 32x32, scalar: 1 invocation(s)
    sin 16x16: 1 invocation(s)
    sin 16x16, scalar: 1 invocation(s)
    sin 32x32: 1 invocation(s)
    sin 32x32, scalar: 1 invocation(s)
    sqrt 16x16: 1 invocation(s)
    sqrt 16x16, scalar: 1 invocation(s)
    sqrt 32x32: 1 invocation(s)
    sqrt 32x32, scalar: 1 invocation(s)
    add 16x64: 1 invocation(s)
    add 16x64, scalar: 1 invocation(s)
    add 32x128: 1 invocation(s)
    add 32x128, scalar: 1 invocation(s)
    div 16x64: 1 invocation(s)
    div 16x64, scalar: 1 invocation(s)
    div 32x128: 1 invocation(s)
    div 32x128, scalar: 1 invocation(s)
    mul 16x64: 1 invocation(s)
    mul 16x64, scalar: 1 invocation(s)
    mul 32x128: 1 invocation(s)
    mul 32x128, scalar: 1 invocation(s)
    x*2 16x64: 1 invocation(s)
    x*2 16x64, scalar: 1 invocation(s)
    x*2 32x128: 1 invocation(s)
    x*2 32x128, scalar: 1 invocation(s)
    x*x+x 16x64: 2 invocation(s)
    x*x+x 16x64, scalar: 2 invocation(s)
    x*x+x 32x128: 2 invocation(s)
    x*x+x 32x128, scalar: 2 invocation(s)
    x+y+z 16x64: 2 invocation(s)
    x+y+z 16x64, scalar: 2 invocation(s)
    x+y+z 32x128: 2 invocation(s)
    x+y+z 32x128, scalar: 2 invocation(s)
    add 16x64, in RAM: 1 invocation(s)
    add 32x128, in RAM: 1 invocation(s)
    div 16x64, in RAM: 1 invocation(s)
    div 32x128, in RAM: 1 invocation(s)
    exp 32x32, in RAM: 1 invocation(s)
    add 16x64, scalar, in RAM: 1 invocation(s)
    softmax over C 16x16: 1 invocation(s)
    softmax over C 32x32: 1 invocation(s)
    sum over C 16x64: 1 invocation(s)
    sum over C 32x128: 1 invocation(s)
    sum over W 16x64: 1 invocation(s)
    transpose 32x32: 1 invocation(s)
    transpose 64x64: 1 invocation(s)
    transpose 32x32, in RAM: 1 invocation(s)
    head split 16x64, 2 heads: 2 invocation(s)
    head split 32x128, 4 heads: 2 invocation(s)
    head split 49x64, 2 heads: 2 invocation(s)
    linear 9x32 k64: 1 invocation(s)
    linear 16x64 k64: 1 invocation(s)
    linear 64x64 k64: 1 invocation(s)
    linear 16x128 k32: 1 invocation(s)
    linear 32x32 k128: 1 invocation(s)
    linear 48x80 k32: 1 invocation(s)
    linear bf16 9x32 k64: 1 invocation(s)
    linear bf16 16x64 k64: 1 invocation(s)
    linear bf16 64x64 k64: 1 invocation(s)
    linear bf16 32x128 k32: 1 invocation(s)
    linear bf16 48x48 k48: 1 invocation(s)
    sdpa bf16 32x32 d32: 1 invocation(s)
    sdpa bf16 64x64 d64: 1 invocation(s)
    sdpa bf16 16x48 d32: 1 invocation(s)
    sdpa bf16 16x16 d32 bk16: 1 invocation(s)
    sdpa bf16 32x64 d64 bk16: 1 invocation(s)
    sdpa bf16 64x32 d32: 1 invocation(s)
    linear 9x32 k64, in RAM: 1 invocation(s)
    linear 32x32 k64, in RAM: 1 invocation(s)
    linear bf16 16x256 k64: 1 invocation(s)
    linear bf16 32x128 k128: 1 invocation(s)
    linear bf16 49x512 k64: 1 invocation(s)
    linear bf16 16x256 k64, in RAM: 1 invocation(s)
    linear bf16 16x64 k64, in RAM: 1 invocation(s)
    linear bf16 64x64 k64, in RAM: 1 invocation(s)
    sdpa bf16 32x32 d32, in RAM: 1 invocation(s)
    sdpa bf16 16x16 d32 bk16, in RAM: 1 invocation(s)
    sdpa bf16 64x64 d64, in RAM: 1 invocation(s)
    sdpa bf16 9x29 d64 masked: 1 invocation(s)
    exp 64x64: 1 invocation(s)
    softmax over C 64x64: 1 invocation(s)
    softmax over W 16x16: 1 invocation(s)
    layer norm over C 16x64: 1 invocation(s)
    rms norm over C 16x64: 1 invocation(s)
    linear 32x96 k48: 1 invocation(s)
    sdpa bf16 41x61 d32: 1 invocation(s)
    layer, bare: 1 invocation(s)
    layer, all flags: 1 invocation(s)
    layer, pre layer norm, rope: 1 invocation(s)
    layer, post rms: 1 invocation(s)
    layer decomposed, bare: 14 invocation(s), 1024 unknown |}]

(* Each innermost loop body is counted apart, trips included: what the
   overlap term pairs. The softmax's three row loops stay three bodies. *)
let%expect_test "innermost loop bodies" =
  List.iter
    (fun name ->
      let c = List.find (fun (c : Cases.t) -> c.Cases.name = name) Cases.all in
      match censuses c with
      | Error e -> print_endline e
      | Ok cs ->
          List.iter
            (fun (x : CE.t) ->
              List.iter
                (fun b ->
                  Fmt.pr "%s:%a@." name
                    Fmt.(
                      list ~sep:nop (fun ppf (cl, v) ->
                          pf ppf " %s=%.0f" (CE.Class.name cl) v))
                    b)
                x.CE.bodies)
            cs)
    [ "div 32x128"; "softmax over C 16x16" ];
  [%expect
    {|
    div 32x128: iteration=256 vec_f32_div=1024 vec_load_dtcm=2048 vec_store_dtcm=1024
    softmax over C 16x16: f32_op=256 iteration=256 load_dtcm=256
    softmax over C 16x16: f32_op=256 int_op=32 iteration=16 vec_f32_op=64 vec_other=256 vec_load_dtcm=64 vec_exp=64
    softmax over C 16x16: iteration=16 vec_f32_div=64 vec_f32_op=64 vec_load_dtcm=64 vec_exp=64 vec_store_dtcm=64 |}]

(* A view call is counted as the helper calls it lowers to (one per row of an
   axis with a base per row, as Mir_lower_contract.view_call emits), with its
   address arithmetic as int ops: a loop of three block calls of four rows is
   three calls on one base, twelve on per-row bases, and the loop's trips are
   still counted as iterations. *)
let%expect_test "view calls are their helper calls" =
  let open Ssa_ir in
  let i64 = Int64.of_int in
  let m = 12 and k = 8 and n = 16 and r = 4 in
  let desc rows =
    Result.get_ok
      (Err.payload
         (Contraction.make ~batch:1L ~m:(i64 rows) ~n:(i64 n) ~k:(i64 k)
            ~a:{ Contraction.Operand.batch = 0L; outer = i64 k; k = 1L }
            ~b:{ Contraction.Operand.batch = 0L; outer = 1L; k = i64 n }
            ~c:{ Contraction.Output.batch = 0L; row = i64 n; col = 1L }
            ~format:Contraction.Element.F32 ~seed:Contraction.Seed.Zero
            ~numerics:Contraction.Numerics.Ordered))
  in
  let buffer id ~h ~w role =
    {
      Ssa_buffer.id = Ssa_id.Buffer.of_int id;
      extents = Expr.Coord.make ~n:1L ~t:1L ~d:1L ~h:(i64 h) ~w:(i64 w) ~c:1L;
      format = Ssa_format.F32;
      role;
    }
  in
  let buffers =
    [
      buffer 0 ~h:m ~w:k Ssa_buffer.Input;
      buffer 1 ~h:k ~w:n Ssa_buffer.Input;
      buffer 2 ~h:m ~w:n Ssa_buffer.Output;
    ]
  in
  let census ~per_row =
    let module B = Ssa_builder in
    let p =
      Err.or_raise ~pp_error:Ssa_verify.pp_error
        (B.program ~buffers (fun bld ->
             ignore
               (B.for_ bld ~lo:(B.index bld 0L)
                  ~hi:(B.index bld (i64 (m / r)))
                  ~init:B.Nil
                  (fun bld blk B.Nil ->
                    let start s = B.index_scale bld (i64 s) blk in
                    let at s =
                      let first = start (r * s) in
                      if per_row then
                        B.Rows
                          (List.init r (fun i ->
                               B.index_add bld first (B.index bld (i64 (i * s)))))
                      else B.Base first
                    in
                    B.view_call bld (desc r)
                      ~a:(Ssa_id.Buffer.of_int 0, at k)
                      ~b:(Ssa_id.Buffer.of_int 1, B.Base (B.index bld 0L))
                      ~c:(Ssa_id.Buffer.of_int 2, at n);
                    B.Nil))))
    in
    CE.program p
  in
  List.iter
    (fun per_row ->
      let c = census ~per_row in
      Fmt.pr "per_row=%b: %a@." per_row CE.pp c)
    [ false; true ];
  [%expect {|
    per_row=false: int_op 42
                   iteration 3
                   contract contract.f32.b1.m4.n16.k8.a0_8_1.b0_1_16.c0_16_1.zero.ordered (dtcm dtcm dtcm) x3
    per_row=true: int_op 174
                  iteration 3
                  contract contract.f32.b1.m1.n16.k8.a0_8_1.b0_1_16.c0_16_1.zero.ordered (dtcm dtcm dtcm) x12 |}]
