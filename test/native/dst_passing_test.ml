(* Destination passing: the evaluator allocates each output from its edge's
   declaration and every arm writes into it. These tests pin the rows a
   mismatched destination produces, that the default executors hand back the
   very destination they were given, and that storage-preserving copies keep raw
   cells and quantization exactly. *)

open Graph_ir

let s1c n = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:n

let sig_ ?quant id shape fmt =
  Tensor_sig.create ~id:(Tensor_id.of_int id) ~name:"" ~shape ~fmt ?quant ()

let f32 = Payload.Fmt Payload.F32
let get_ok = function Ok x -> x | Error _ -> Fmt.failwith "unexpected error"
let pp_dst_error ppf e = Tensor.pp_dst_error ppf e

let show_write label r =
  match Err.payload r with
  | Ok () -> Fmt.pr "%s: ok@." label
  | Error e -> Fmt.pr "%s: %a@." label pp_dst_error e

let create sg = get_ok (Err.payload (Tensor.create_of_sig sg))

let%expect_test "writers refuse a destination of another format" =
  let shape = s1c 3 in
  let dst_of fmt = create (sig_ 0 shape (Payload.Fmt fmt)) in
  show_write "float into f32"
    (Tensor.write_float (dst_of Payload.F32) (fun _ -> 1.));
  show_write "float into f16"
    (Tensor.write_float (dst_of Payload.F16) (fun _ -> 1.));
  show_write "float into i64"
    (Tensor.write_float (dst_of Payload.I64) (fun _ -> 1.));
  show_write "float into bool"
    (Tensor.write_float (dst_of Payload.Bool) (fun _ -> 1.));
  show_write "i64 into i64"
    (Tensor.write_i64 (dst_of Payload.I64) (fun _ -> 1L));
  show_write "i64 into f32"
    (Tensor.write_i64 (dst_of Payload.F32) (fun _ -> 1L));
  show_write "bool into bool"
    (Tensor.write_bool (dst_of Payload.Bool) (fun _ -> true));
  show_write "bool into f32"
    (Tensor.write_bool (dst_of Payload.F32) (fun _ -> true));
  [%expect
    {|
    float into f32: ok
    float into f16: ok
    float into i64: destination format i64 cannot take float writes
    float into bool: destination format bool cannot take float writes
    i64 into i64: ok
    i64 into f32: destination format f32 cannot take i64 writes
    bool into bool: ok
    bool into f32: destination format f32 cannot take bool writes |}]

let%expect_test "create_of_sig builds what the signature declares" =
  let q = Quant.per_tensor ~scale:0.5 ~zero_point:3 in
  let describe sg =
    match Err.payload (Tensor.create_of_sig sg) with
    | Ok (Tensor.Tensor t) ->
        Fmt.pr "%s %s@."
          (Payload.fmt_name t.Tensor.payload.Payload.fmt)
          (match t.Tensor.payload.Payload.quant with
          | Payload.Quant _ -> "quantized"
          | Payload.No_quant -> "plain")
    | Error (`Quant_missing id) ->
        Fmt.pr "quant missing for %a@." Tensor_id.pp id
  in
  describe (sig_ 1 (s1c 2) f32);
  describe (sig_ ~quant:q 2 (s1c 2) (Payload.Fmt Payload.I8));
  describe (sig_ ~quant:q 3 (s1c 2) (Payload.Fmt Payload.I16));
  describe (sig_ 4 (s1c 2) (Payload.Fmt Payload.I8));
  describe (sig_ 5 (s1c 2) (Payload.Fmt Payload.I64));
  [%expect
    {|
    f32 plain
    i8 quantized
    i16 quantized
    quant missing for t4
    i64 plain |}]

(* ---- the default executors return the destination they were given --------- *)

let inputs_of (g : graph) tensors =
  List.map2 (fun id t -> (id, t)) g.Graph.inputs tensors

let tensor_of_sig (sg : Tensor_sig.t) =
  Tensor.materialize sg.Tensor_sig.shape (fun c ->
      float_of_int ((Vec6.offset sg.Tensor_sig.shape c :> int) mod 5) /. 4.)

(* Counts, over one run, results that are physically the destination and results
   that are not, for each of the three executor seams. *)
let identity_run (g : graph) =
  let same = ref 0 and other = ref 0 in
  let note dst r =
    match Err.payload r with
    | Ok t -> if t == dst then incr same else incr other
    | Error _ -> ()
  in
  let node_executor =
    {
      Node_executor.run =
        (fun _ _ ~output:_ ~out_shape:_ ~operands:_ ~dst ~direct ->
          let r = direct ~dst in
          note dst r;
          r);
    }
  in
  let region_executor : Region_executor.t =
   fun ?counters ~dst lowered ~env ~bindings ->
    let r = Region_executor.default ?counters ~dst lowered ~env ~bindings in
    note dst r;
    r
  in
  let region_group_executor : Region_executor.group =
   fun ?counters ~dsts lowered ~env ~bindings ->
    let r =
      Region_executor.default_group ?counters ~dsts lowered ~env ~bindings
    in
    (match Err.payload r with
    | Ok results ->
        List.iter
          (fun (o, t) ->
            if t == List.assoc o dsts then incr same else incr other)
          results
    | Error _ -> ());
    r
  in
  let bound kind =
    List.filter_map
      (fun id ->
        if input_kind g id = kind then
          Some (id, tensor_of_sig (Tensor_id.Map.find id g.Graph.tensors))
        else None)
      g.Graph.inputs
  in
  let ok =
    Result.is_ok
      (Err.payload
         (Eval_direct.run ~node_executor ~region_executor ~region_group_executor
            ~constants:(bound Input.Constant) g ~inputs:(bound Input.Input)))
  in
  (ok, !same, !other)

let%expect_test "default executors hand back the destination" =
  let softmax =
    Graph_fixtures.build "softmax"
      Graph_builder.(
        let* x = input ~shape:(s1c 4) () in
        softmax { Reduce.Softmax.axis = Axis.C } x)
  in
  List.iter
    (fun (name, g) ->
      let ok, same, other = identity_run g in
      Fmt.pr "%s: ran %b, %d destination, %d other@." name ok same other)
    [
      ("pixel ops", Graph_fixtures.residual ());
      ("region op", softmax);
      ("lstm group", Lstm_graph_test.graph);
    ];
  [%expect
    {|
    pixel ops: ran true, 3 destination, 0 other
    region op: ran true, 1 destination, 0 other
    lstm group: ran true, 3 destination, 0 other |}]

(* ---- storage-preserving copies keep raw cells and quantization ------------ *)

let quantized_tensor fmt shape cell =
  let n = (Vec6.numel shape :> int) in
  let q = Quant.per_tensor ~scale:0.25 ~zero_point:2 in
  match fmt with
  | `I8 ->
      let data = Bigarray.(Array1.create int8_signed c_layout n) in
      for i = 0 to n - 1 do
        data.{i} <- cell i
      done;
      Tensor.Tensor
        {
          shape;
          payload = { Payload.fmt = Payload.I8; quant = Payload.Quant q; data };
        }
  | `I16 ->
      let data = Bigarray.(Array1.create int16_signed c_layout n) in
      for i = 0 to n - 1 do
        data.{i} <- cell i
      done;
      Tensor.Tensor
        {
          shape;
          payload = { Payload.fmt = Payload.I16; quant = Payload.Quant q; data };
        }

(* Raw cells and the quantization, exactly. *)
let cells_of : type e b q. (e, b, q) Tensor.t -> int list =
 fun t ->
  let n = (Vec6.numel t.Tensor.shape :> int) in
  match t.Tensor.payload.Payload.fmt with
  | Payload.I8 -> List.init n (fun i -> t.Tensor.payload.Payload.data.{i})
  | Payload.I16 -> List.init n (fun i -> t.Tensor.payload.Payload.data.{i})
  | _ -> []

let raw (Tensor.Tensor t) =
  let quant =
    match t.Tensor.payload.Payload.quant with
    | Payload.Quant q -> Some q
    | Payload.No_quant -> None
  in
  (Payload.fmt_name t.Tensor.payload.Payload.fmt, cells_of t, quant)

let same_raw a b =
  let fa, ca, qa = raw a and fb, cb, qb = raw b in
  String.equal fa fb && ca = cb && Option.equal Quant.equal qa qb

let%expect_test "quantized unbind and split_with_sizes copy raw cells and quant"
    =
  let sizes = [ Dim.extent 1; Dim.extent 2 ] in
  let shape = Vec6.shape ~n:2 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3 in
  List.iter
    (fun fmt ->
      let src = quantized_tensor fmt shape (fun i -> (i * 7) - 20) in
      let fmt_name = match fmt with `I8 -> "i8" | `I16 -> "i16" in
      let x_fmt =
        match fmt with
        | `I8 -> Payload.Fmt Payload.I8
        | `I16 -> Payload.Fmt Payload.I16
      in
      let (Tensor.Tensor s) = src in
      let quant =
        match s.Tensor.payload.Payload.quant with
        | Payload.Quant q -> q
        | Payload.No_quant -> assert false
      in
      let graph name outputs =
        Graph_fixtures.buildn name (outputs quant x_fmt)
      in
      (* Unbind along N: two slices. *)
      let unbind =
        graph "unbind" (fun quant x_fmt ->
            Graph_builder.(
              let* x = input ~shape ~fmt:x_fmt ~quant () in
              unbind { Split.Unbind.axis = Axis.N } x))
      in
      let split =
        graph "split" (fun quant x_fmt ->
            Graph_builder.(
              let* x = input ~shape ~fmt:x_fmt ~quant () in
              split_with_sizes { Split.Split_with_sizes.axis = Axis.C; sizes } x))
      in
      let check name (g : graph) reference =
        match
          Err.payload
            (Eval_direct.run g ~inputs:[ (List.hd g.Graph.inputs, src) ])
        with
        | Error _ -> Fmt.pr "%s %s: run failed@." fmt_name name
        | Ok env ->
            List.iteri
              (fun i id ->
                let got = Tensor_id.Map.find id env in
                let out_shape =
                  (Tensor_id.Map.find id g.Graph.tensors).Tensor_sig.shape
                in
                let want = reference i out_shape in
                Fmt.pr "%s %s output %d: %s@." fmt_name name i
                  (if same_raw got want then "same raw cells and quantization"
                   else "DIFFERS"))
              g.Graph.outputs
      in
      check "unbind" unbind (fun i out_shape ->
          Tensor.unbind src ~axis:Axis.N ~output:(Output_ordinal.of_int i)
            ~shape:out_shape);
      check "split" split (fun i out_shape ->
          Tensor.split_with_sizes src ~axis:Axis.C
            ~offset:
              (Split.Split_with_sizes.offset_of
                 ~output:(Output_ordinal.of_int i) sizes)
            ~shape:out_shape))
    [ `I8; `I16 ];
  [%expect
    {|
    i8 unbind output 0: same raw cells and quantization
    i8 unbind output 1: same raw cells and quantization
    i8 split output 0: same raw cells and quantization
    i8 split output 1: same raw cells and quantization
    i16 unbind output 0: same raw cells and quantization
    i16 unbind output 1: same raw cells and quantization
    i16 split output 0: same raw cells and quantization
    i16 split output 1: same raw cells and quantization |}]

let%expect_test "a storage copy refuses a destination of another quantization" =
  let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2 in
  let src = quantized_tensor `I8 shape (fun i -> i) in
  let other_q = Quant.per_tensor ~scale:1. ~zero_point:0 in
  let dst_q = create (sig_ ~quant:other_q 0 shape (Payload.Fmt Payload.I8)) in
  let dst_f = create (sig_ 1 shape (Payload.Fmt Payload.I16 |> fun _ -> f32)) in
  let copy dst =
    let (Tensor.Tensor s) = src in
    Tensor.copy_cells_into s.Tensor.payload dst ~shape ~src_shape:shape
      ~source_coord:(fun c -> c)
  in
  show_write "other quantization" (copy dst_q);
  show_write "other format" (copy dst_f);
  [%expect
    {|
    other quantization: destination quantization differs from the source's: a storage copy never re-quantizes
    other format: destination format f32 cannot take same-format copy writes |}]
