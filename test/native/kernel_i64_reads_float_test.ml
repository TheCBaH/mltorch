(* An int64 [Kernel.Value_i64.t] reading a float input or a COMPUTED float value
   (the tracker's D10): mvitv2's [add.Tensor -> _to_copy(Long)] shape. The two
   carriers may now read each other in either direction, so neither list is
   evaluated "first" -- these fixtures pin the answers, the reachability rule
   (an int64 entry is a root, so a float only it reads is live) and the
   position rule that keeps the combined graph acyclic. *)

let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3
let tid = Tensor_id.of_int
let f32 = Payload.Fmt Payload.F32
let i64 = Payload.Fmt Payload.I64
let sg fmt id = Tensor_sig.create ~id:(tid id) ~name:"" ~shape ~fmt ()
let here = Expr_bridge.coord_of_vec6 Symbolic.out_vec
let load id = Expr.Value.load (Expr_bridge.source_of_id (tid id)) here
let load_i64 id = Expr.Value.i64_load (Expr_bridge.source_of_id (tid id)) here

let float_value id body =
  {
    Kernel.Value.id = tid id;
    sg = sg f32 id;
    computation = Region_group.Ref.Solo (Region_program.pixel body);
    result = Kernel.Result_conversion.Round_f32;
  }

let value_i64 id pixel = { Kernel.Value_i64.id = tid id; sg = sg i64 id; pixel }
let input id = { Kernel.Input.id = tid id; sg = sg f32 id; binding = Caller }
let pp_kernel = Core.Pretty.err_result ~ok:(Fmt.any "ok") ~error:Kernel.pp_error

let x =
  let cells = [| 1.9; -2.7; 3. |] in
  Tensor.materialize shape (fun c -> cells.(Dim.to_int (Vec6.get c Axis.C)))

let bind id = if Tensor_id.equal id (tid 0) then Some x else None

let run k =
  Err.or_raise ~pp_error:Kernel_eval.pp_error (Kernel_eval.run k ~bind)

let ints t =
  List.init 3 (fun c ->
      match Tensor.read_i64_at6 t (function Axis.C -> c | _ -> 0) with
      | Ok v -> Int64.to_string v
      | Error _ -> "not i64")
  |> String.concat ","

let floats t =
  List.init 3 (fun c -> Tensor.read_at_raw t (function Axis.C -> c | _ -> 0))
  |> List.map (Printf.sprintf "%g")
  |> String.concat ","

(* t0 (f32 input) -> t1 (i64, truncates toward zero) *)
let%expect_test "an int64 value reads a float input" =
  let k =
    Kernel.create
      ~inputs:[ input 0 ]
      ~values_i64:[ value_i64 1 (Expr.Value.float_to_i64 (load 0)) ]
      ~values:[] ~outputs:[] ()
    |> Err.or_raise ~pp_error:Kernel.pp_error
  in
  Fmt.pr "t1 = %s@." (ints (Tensor_id.Map.find (tid 1) (run k)));
  [%expect {| t1 = 1,-2,3 |}]

(* t0 -> t1 = t0 + 0.5 (float stage) -> t2 = trunc t1 (i64) -> t3 = t2 * 2 (float).
   t1 is read only by the int64 value, yet must be live and stored. *)
let chain =
  Kernel.create
    ~inputs:[ input 0 ]
    ~values_i64:[ value_i64 2 (Expr.Value.float_to_i64 (load 1)) ]
    ~values:
      [
        float_value 1 (Expr.Value.add (load 0) (Expr.Value.const 0.5));
        float_value 3
          (Expr.Value.mul
             (Expr.Value.i64_to_float (load_i64 2))
             (Expr.Value.const 2.));
      ]
    ~outputs:[ tid 3 ]
    ()
  |> Err.or_raise ~pp_error:Kernel.pp_error

let%expect_test
    "an int64 value reads a computed float, and a float reads it back" =
  let r = run chain in
  Fmt.pr "t2 = %s@.t3 = %s@."
    (ints (Tensor_id.Map.find (tid 2) r))
    (floats (Tensor_id.Map.find (tid 3) r));
  [%expect {|
    t2 = 2,-2,3
    t3 = 4,-4,6 |}]

let%expect_test "on-demand and planned readings agree with the stored one" =
  let plan, _ = Fusion_plan.plan chain in
  let r =
    Err.or_raise ~pp_error:Kernel_eval.pp_error
      (Kernel_eval.run_plan plan ~bind)
  in
  Fmt.pr "planned t3 = %s@." (floats (Tensor_id.Map.find (tid 3) r));
  let at c =
    Err.or_raise ~pp_error:Kernel_eval.pp_error
      (Kernel_eval.value_at chain ~bind (tid 3)
         (Expr.Coord.make ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c))
  in
  Fmt.pr "value_at t3 = %g,%g,%g@." (at 0) (at 1) (at 2);
  [%expect {|
    planned t3 = 4,-4,6
    value_at t3 = 4,-4,6 |}]

(* f1 reads the int64 V, and V reads f2, which is defined AFTER f1: a forward
   reference through the int64 side, the shape a cycle would take. *)
let%expect_test "a float reading an int64 that reads a LATER float is rejected"
    =
  Fmt.pr "%a@." pp_kernel
    (Kernel.create
       ~inputs:[ input 0 ]
       ~values_i64:[ value_i64 9 (Expr.Value.float_to_i64 (load 2)) ]
       ~values:
         [
           float_value 1 (Expr.Value.i64_to_float (load_i64 9));
           float_value 2 (Expr.Value.add (load 0) (Expr.Value.const 1.));
         ]
       ~outputs:[ tid 1 ]
       ());
  [%expect {| t1 depends on later value t9 |}]

let%expect_test "an int64 value reading an unknown id is still rejected" =
  Fmt.pr "%a@." pp_kernel
    (Kernel.create ~inputs:[]
       ~values_i64:[ value_i64 1 (Expr.Value.float_to_i64 (load 7)) ]
       ~values:[] ~outputs:[] ());
  [%expect
    {| t1: an int64 value may only read an input, a float value or an earlier int64 value |}]

(* t1 = t0 + 0.5 is read by the float t3 (which fusion may inline) AND by the
   int64 t2. The int64 read must still find a real t1 even when the plan
   virtualizes t1 for t3. *)
let shared =
  Kernel.create
    ~inputs:[ input 0 ]
    ~values_i64:[ value_i64 2 (Expr.Value.float_to_i64 (load 1)) ]
    ~values:
      [
        float_value 1 (Expr.Value.add (load 0) (Expr.Value.const 0.5));
        float_value 3 (Expr.Value.mul (load 1) (Expr.Value.const 2.));
      ]
    ~outputs:[ tid 3 ]
    ()
  |> Err.or_raise ~pp_error:Kernel.pp_error

let%expect_test "a float virtualized for one reader is still real for the int64"
    =
  let plan, _ = Fusion_plan.plan shared in
  Fmt.pr "virtual uses: %d@."
    (Kernel.Use.Set.cardinal plan.Fusion_plan.virtual_uses);
  let r =
    Err.or_raise ~pp_error:Kernel_eval.pp_error
      (Kernel_eval.run_plan plan ~bind)
  in
  Fmt.pr "t2 = %s@.t3 = %s@."
    (ints (Tensor_id.Map.find (tid 2) r))
    (floats (Tensor_id.Map.find (tid 3) r));
  [%expect {|
    virtual uses: 1
    t2 = 2,-2,3
    t3 = 4.8,-4.4,7 |}]

(* A typed int64 reduction through the Kernel: t1 = sum over c < 3 of
   trunc(t0[C=c]) as an exact int64 accumulator, reading the float input. With
   t0 = [1.9, -2.7, 3.0] the truncations are 1, -2, 3 and the sum is 2. *)
let%expect_test "an int64 value sums truncated floats with an exact accumulator"
    =
  let one = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let sum =
    Expr.Builder.run
      (Expr.Builder.i64_sum ~lo:Expr.Index.zero ~hi:(Expr.Index.const 3)
         (fun r ->
           Expr.Builder.return
             (Expr.Value.float_to_i64
                (Expr.Value.load
                   (Expr_bridge.source_of_id (tid 0))
                   (Expr.Coord.set here Expr.Axis.C r)))))
  in
  let k =
    Kernel.create
      ~inputs:[ input 0 ]
      ~values_i64:
        [
          {
            Kernel.Value_i64.id = tid 1;
            sg = Tensor_sig.create ~id:(tid 1) ~name:"" ~shape:one ~fmt:i64 ();
            pixel = sum;
          };
        ]
      ~values:[] ~outputs:[] ()
    |> Err.or_raise ~pp_error:Kernel.pp_error
  in
  let r = run k in
  let t = Tensor_id.Map.find (tid 1) r in
  Fmt.pr "t1 = %a@." Tensor.pp t;
  [%expect {| t1 = tensor i64 [C=1] {2} |}]

(* Typed filled inputs (P2.3): an I64 input filled with an int64 stays exact past
   2^53, and the fill is checked against its signature. t0 is filled with
   2^53 + 1 (a float would round it), t1 = t0 + 1 reads it exactly. *)
let%expect_test "an int64 filled input is exact and its format is checked" =
  let big = 9_007_199_254_740_993L in
  let filled binding fmt =
    { Kernel.Input.id = tid 0; sg = sg fmt 0; binding }
  in
  let make binding fmt =
    Kernel.create
      ~inputs:[ filled binding fmt ]
      ~values_i64:
        [
          value_i64 1
            (Expr.Value.i64_add (load_i64 0) (Expr.Value.i64_const 1L));
        ]
      ~values:[] ~outputs:[] ()
  in
  let k =
    make (Kernel.Binding.Filled_i64 big) i64
    |> Err.or_raise ~pp_error:Kernel.pp_error
  in
  let r =
    Err.or_raise ~pp_error:Kernel_eval.pp_error
      (Kernel_eval.run k ~bind:(fun _ -> None))
  in
  Fmt.pr "t1 = %s@." (ints (Tensor_id.Map.find (tid 1) r));
  Fmt.pr "filled i64 on an f32 input: %a@." pp_kernel
    (make (Kernel.Binding.Filled_i64 big) f32);
  Fmt.pr "float fill on an i64 input: %a@." pp_kernel
    (make (Kernel.Binding.Filled 1.) i64);
  [%expect
    {|
    t1 = 9007199254740994,9007199254740994,9007199254740994
    filled i64 on an f32 input: t0: an int64 value must be i64 and unquantized, got f32
    float fill on an i64 input: t0: a filled input must be f32 or bool and unquantized, got i64 |}]
