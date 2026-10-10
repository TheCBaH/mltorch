module P = Pytorch_types
module S = Schema_runtime.String_map

let float_tensor values shape =
  let data =
    Bigarray.Array1.of_array Bigarray.float32 Bigarray.c_layout values
  in
  Aten_tensor.of_bigarray Aten_dtype.float32 data shape

let long_tensor values shape =
  let data = Bigarray.Array1.of_array Bigarray.int64 Bigarray.c_layout values in
  Aten_tensor.of_bigarray Aten_dtype.int64 data shape

let tensor name = P.TensorArgument.make name
let arg name value = P.NamedArgument.make name value None

let node indices =
  P.Node.make "torch.ops.aten.index.Tensor"
    [ arg "self" (P.Argument.Tensor (tensor "x")); arg "indices" indices ]
    [ P.Argument.Tensor (tensor "out") ]
    S.empty None None

let show env indices =
  match Err.payload (Interp_dispatch.dispatch env (node indices)) with
  | Error error -> Fmt.pr "%a@." Interp.pp_error error
  | Ok env ->
      let t = S.find "out" env |> Aten_tensor.materialize_for_raw_read in
      let data = Option.get (Aten_tensor.data Aten_dtype.float32 t) in
      Fmt.pr "[%s] [%s]@."
        (String.concat ";"
           (Array.to_list (Aten_tensor.shape t) |> List.map string_of_int))
        (String.concat ";"
           (List.init (Bigarray.Array1.dim data) (fun i ->
                Printf.sprintf "%g" data.{i})))

let%expect_test
    "C++ advanced indexing keeps absent axes and handles all-live encoding" =
  let env =
    S.empty
    |> S.add "x" (float_tensor [| 1.; 2.; 3.; 4.; 5.; 6. |] [ 2; 3 ])
    |> S.add "cols" (long_tensor [| 2L; 0L |] [ 2 ])
    |> S.add "rows" (long_tensor [| 1L; 0L |] [ 2 ])
  in
  show env
    (P.Argument.Optional_tensors
       [
         P.OptionalTensorArgument.None false;
         P.OptionalTensorArgument.Tensor (tensor "cols");
       ]);
  show env (P.Argument.Tensors [ tensor "rows"; tensor "cols" ]);
  [%expect {|
    [2;2] [3;1;6;4]
    [2] [6;1] |}]

let%expect_test "C++ advanced indices broadcast independently" =
  let env =
    S.empty
    |> S.add "x" (float_tensor [| 1.; 2.; 3.; 4.; 5.; 6. |] [ 2; 3 ])
    |> S.add "cols" (long_tensor [| 2L; 0L |] [ 1; 2 ])
    |> S.add "rows" (long_tensor [| 1L; 0L |] [ 2; 1 ])
  in
  show env (P.Argument.Tensors [ tensor "rows"; tensor "cols" ]);
  [%expect {| [2;2] [6;4;3;1] |}]

let%expect_test "C++ indexing bounds errors return through the ABI" =
  let env =
    S.empty
    |> S.add "x" (float_tensor [| 1.; 2.; 3.; 4.; 5.; 6. |] [ 2; 3 ])
    |> S.add "rows" (long_tensor [| 99L |] [ 1 ])
  in
  (try
     ignore
       (Interp_dispatch.dispatch env
          (node (P.Argument.Tensors [ tensor "rows" ])));
     print_endline "unexpected pass"
   with Aten_tensor.Error error ->
     Fmt.pr "bounds error retained=%b@."
       (String.starts_with ~prefix:"index" error));
  [%expect {| bounds error retained=true |}]
