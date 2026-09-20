(* [Max_dim4]: the paired value/index reduction along one axis. Hand-computed
   answers on Direct pin the tie rule (the smallest position wins) and the
   keepdim shapes; the Native comparison pins the lowering, on values with ties
   and a NaN, since the values fold with the same predicate as the index. *)

open Native4d

let s = Fixtures.s
let show name t = Fmt.pr "%s = %a@." name Tensor.pp t

let%expect_test "ties resolve to the first position, keepdim both ways" =
  let shape = Shape4.of_ints ~n:1 ~h:2 ~w:4 ~c:1 in
  let x =
    Tensor.materialize (Shape4.to_vec6 shape) (fun c ->
        let h = Dim.to_int (Vec6.get c Axis.H)
        and w = Dim.to_int (Vec6.get c Axis.W) in
        [| [| 1.; 3.; 3.; 2. |]; [| 5.; 5.; 4.; 5. |] |].(h).(w))
  in
  List.iter
    (fun keepdim ->
      let g =
        Builder.build ~outputs:Fun.id
          Builder.(
            let* x = input ~shape () in
            max_dim4 { Ops4_max_dim.Max_dim4.axis = Axis4.W; keepdim } x)
        |> Err.or_raise ~pp_error:Builder.pp_error
      in
      let out =
        Eval_direct4.run g ~constants:[]
          ~inputs:(List.combine g.Graph.Graph.inputs [ x ])
        |> Err.or_raise ~pp_error:Eval_direct4.pp_error
      in
      match g.Graph.Graph.outputs with
      | [ v; i ] ->
          show (Fmt.str "keepdim=%b value" keepdim) (Tensor_id.Map.find v out);
          show (Fmt.str "keepdim=%b index" keepdim) (Tensor_id.Map.find i out)
      | _ -> assert false)
    [ true; false ];
  [%expect
    {|
    keepdim=true value = tensor f32 [H=2 W=1 C=1] {3, 5}
    keepdim=true index = tensor i64 [H=2 W=1 C=1] {1, 0}
    keepdim=false value = tensor f32 [W=2 C=1] {3, 5}
    keepdim=false index = tensor i64 [W=2 C=1] {1, 0} |}]

(* Ties everywhere, and a NaN in one lane. *)
let tied shape =
  let i = ref (-1) in
  Tensor.materialize shape (fun _ ->
      incr i;
      if !i = 13 then Float.nan else float_of_int (!i * 7 mod 5))

let native_graph ~axis ~keepdim =
  Graph_builder.build ~name:"max_dim" ~outputs:Fun.id
    (let open Graph_builder in
     let* x = input ~shape:(s 1 1 1 3 4 5) () in
     let* v, i = max_dim { Reduce.MaxDim.axis; keepdim } x in
     Graph_builder.return [ v; i ])
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

(* Bitwise, not [=]: a NaN is not equal to itself. *)
let agrees ~fill name g =
  match Snapshot.create g with
  | Error _ -> Fmt.pr "%s: snapshot failed@." name
  | Ok (Snapshot.Pack src) -> (
      match Lower.convert src with
      | Error e -> Fmt.pr "%s: %a@." name Error.pp (Err.Error.kind e)
      | Ok (Lower.Pack r) ->
          let dst = Lower.graph r in
          let xs = List.map fill [ s 1 1 1 3 4 5 ] in
          let native =
            Eval_direct.run g
              ~inputs:(List.combine g.Graph_common.Graph.inputs xs)
            |> Err.or_raise ~pp_error:Eval_direct.pp_error
          in
          let four =
            Eval_direct4.run dst ~constants:[]
              ~inputs:(List.combine dst.Graph_common.Graph.inputs xs)
            |> Err.or_raise ~pp_error:Eval_direct4.pp_error
          in
          Fmt.pr "%s: %b@." name
            (List.for_all2
               (fun a b ->
                 Tensor.equal_bits
                   (Tensor_id.Map.find a native)
                   (Tensor_id.Map.find b four))
               g.Graph_common.Graph.outputs dst.Graph_common.Graph.outputs))

let%expect_test "lowering agrees with Native on ties and NaN" =
  List.iter
    (fun axis ->
      List.iter
        (fun keepdim ->
          agrees ~fill:tied
            (Fmt.str "%a keepdim=%b" Axis.pp axis keepdim)
            (native_graph ~axis ~keepdim))
        [ true; false ])
    [ Axis.H; Axis.W; Axis.C ];
  [%expect
    {|
    H keepdim=true: true
    H keepdim=false: true
    W keepdim=true: true
    W keepdim=false: true
    C keepdim=true: true
    C keepdim=false: true |}]

let%expect_test "an axis outside the dialect is refused" =
  agrees ~fill:tied "D" (native_graph ~axis:Axis.D ~keepdim:true);
  [%expect {| D: node n0: axis D is outside the N/H/W/C dialect |}]
