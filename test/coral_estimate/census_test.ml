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
  match
    M.prepare ~contractions:c.Cases.contractions ~bf16_pairs:c.Cases.bf16_pairs
      ?attention:c.Cases.attention ?layer:c.Cases.layer
      ~pipeline:(Cases.pipeline c)
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
    softmax over C 16x16: f32_op=256 iteration=256 local=256 f32_exp=256 load_dtcm=256
    softmax over C 16x16: f32_div=256 f32_op=256 iteration=256 local=512 f32_exp=256 load_dtcm=256 store_dtcm=256 |}]
