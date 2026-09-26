(* A Bool value inside a multi-output [Region_group]. Two sibling emitters share
   one local (W + 1, so 1..4) and one evaluation; the F32 sibling stores it
   rounded, the Bool sibling stores [Nonzero_bool] of a raw body (local - 2, so
   -1, 0, 1, 2 -> true, false, true, true). Each member must get its OWN
   conversion and storage format from its own [Kernel.Value.t], not the group's
   first member's. *)

let max_size = 64
let max_depth = 16
let canonical_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:4 ~c:1
let out_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:4 ~c:1

let whole axes =
  match Region_partition.of_whole_axes axes with
  | Ok p -> p
  | Error _ -> assert false

let shared_local () =
  let id, _ =
    Expr.Builder.run_from Expr.Builder.initial Expr.Builder.fresh_local
  in
  Region_local.scalar ~id
    ~value:
      (Expr.Value.add
         (Expr.Value.value_of_index
            (Expr.Index.of_position (Expr.Index.output Expr.Axis.W)))
         (Expr.Value.const 1.))

let group =
  let local = shared_local () in
  let read = Expr.Value.local local.Region_local.id in
  let emitter output : Region_group.Emitter.t =
    {
      output_shape = out_shape;
      partition = whole [ Expr.Axis.H ];
      key_axes = [ (Expr.Axis.W, Expr.Axis.W) ];
      output;
    }
  in
  Region_group.create ~max_size ~max_depth ~canonical_shape ~locals:[ local ]
    ~emitters:
      [ emitter read; emitter (Expr.Value.sub read (Expr.Value.const 2.)) ]
  |> Err.or_raise ~pp_error:Region_group.pp_error

let value id ordinal fmt result =
  let tid = Tensor_id.of_int id in
  {
    Kernel.Value.id = tid;
    sg = Tensor_sig.create ~id:tid ~name:"" ~shape:out_shape ~fmt ();
    computation =
      Region_group.Ref.Grouped (group, Region_group.Ordinal.of_int ordinal);
    result;
  }

let kernel =
  Kernel.create ~inputs:[]
    ~values:
      [
        value 1 0 (Payload.Fmt Payload.F32) Kernel.Result_conversion.Round_f32;
        value 2 1 (Payload.Fmt Payload.Bool)
          Kernel.Result_conversion.Nonzero_bool;
      ]
    ~outputs:[ Tensor_id.of_int 1; Tensor_id.of_int 2 ]
    ()
  |> Err.or_raise ~pp_error:Kernel.pp_error

let%expect_test "each group member stores in its own format" =
  let results =
    Err.or_raise ~pp_error:Kernel_eval.pp_error
      (Kernel_eval.run kernel ~bind:(fun _ -> None))
  in
  List.iter
    (fun id ->
      let t = Tensor_id.Map.find (Tensor_id.of_int id) results in
      let (Tensor.Tensor p) = t in
      Format.printf "t%d (%s) w=0..3 at h=2: %a@." id
        (Payload.fmt_name p.Tensor.payload.Payload.fmt)
        (Fmt.list ~sep:(Fmt.any ",") Fmt.float)
        (List.init 4 (fun w ->
             Tensor.read t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:2 ~w ~c:0))))
    [ 1; 2 ];
  [%expect
    {|
    t1 (f32) w=0..3 at h=2: 1,2,3,4
    t2 (bool) w=0..3 at h=2: 1,0,1,1 |}]

let%expect_test "a second run leaves the first run's results intact" =
  let run () =
    Err.or_raise ~pp_error:Kernel_eval.pp_error
      (Kernel_eval.run kernel ~bind:(fun _ -> None))
  in
  let first = run () in
  let before =
    Format.asprintf "%a" Tensor.pp
      (Tensor_id.Map.find (Tensor_id.of_int 2) first)
  in
  ignore (run ());
  let after =
    Format.asprintf "%a" Tensor.pp
      (Tensor_id.Map.find (Tensor_id.of_int 2) first)
  in
  Format.printf "unchanged: %b@." (String.equal before after);
  [%expect {| unchanged: true |}]
