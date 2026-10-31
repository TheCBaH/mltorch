open Ssa_bridge
open Ssa_ir
module Lf = Loop_ir_test.Loop_fixtures
module T = Ssa_ir_test.Ssa_norm_test
module N = Norm_block
module P = Attention_block.Precision

(* A row normalization's loops, narrowed, through generated C: the output is
   [Norm_block.eval]'s, bit for bit. *)

let case ~kind ~precision ~weight ~bias ~rows ~width =
  let d = T.desc ~kind ~precision ~weight ~bias ~rows ~width in
  let p = Ssa_precision.to_f32 ~exact:true (T.program d) in
  let ((x, w, b) as cells) = T.inputs d in
  let want = T.reference d cells in
  let kernel, sites =
    match Ssa_c.kernel ~name:Loop_c_exec.kernel_name p with
    | Ok k -> k
    | Error e -> Fmt.failwith "%a" Ssa_c.pp_error e
  in
  let loop =
    Lf.program
      ~buffers:(List.map Loop_of_ssa.loop_buffer (Ssa_c.arguments p))
      []
  in
  let inputs =
    [ (0, (rows, width + T.pad, x)); (1, (1, width, w)); (2, (1, width, b)) ]
  in
  let bind id =
    List.find_map
      (fun (n, (h, w, cs)) ->
        if Tensor_id.equal id (Lf.tid n) then
          let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h ~w ~c:1 in
          Some
            (Lf.f32_tensor shape (fun c -> cs.((Vec6.offset shape c :> int))))
        else None)
      inputs
  in
  match Err.payload (Loop_c_exec.exec_kernel ~kernel ~sites loop ~bind) with
  | Error e -> Fmt.failwith "%a" Loop_c_exec.pp_error e
  | Ok outputs ->
      let got =
        match Tensor_id.Map.find_opt (Lf.tid 3) outputs with
        | Some t ->
            Array.init (rows * width) (fun i ->
                Tensor.read t
                  (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:(i / width) ~w:(i mod width)
                     ~c:0))
        | None -> [||]
      in
      T.bitwise got want

let%expect_test "a norm's loops through C equal the reference" =
  List.iter
    (fun kind ->
      List.iter
        (fun precision ->
          let all =
            List.for_all
              (fun (rows, width) ->
                List.for_all
                  (fun (weight, bias) ->
                    (bias && kind = N.Kind.Rms)
                    || case ~kind ~precision ~weight ~bias ~rows ~width)
                  T.affines)
              (T.shapes @ [ (32, 17) ])
          in
          Fmt.pr "%s %s: C equals the reference: %b@." (N.Kind.name kind)
            (P.name precision) all)
        [ P.F32_ordered; P.F32_fused ])
    [ N.Kind.Layer; N.Kind.Rms ];
  [%expect
    {|
    layer f32_ordered: C equals the reference: true
    layer f32_fused: C equals the reference: true
    rms f32_ordered: C equals the reference: true
    rms f32_fused: C equals the reference: true |}]
