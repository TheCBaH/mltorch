(* A Bool tensor read from a .pt2 archive (a captured constant or an input): its
   signature format is Bool, its bytes read as logical values (a nonzero byte is
   true) and are stored canonical, so a noncanonical byte in the archive cannot
   leak into a consumer. *)

let bool_tensor bytes =
  let data =
    Bytes.init (List.length bytes) (fun i -> Char.chr (List.nth bytes i))
  in
  Pt2_tensor.
    {
      dtype = Pt2_dtype.Bool;
      sizes = [ List.length bytes ];
      strides = [ 1 ];
      storage_offset = 0;
      data;
    }

let%expect_test
    "a Bool archive tensor keeps its format and reads bytes as logic" =
  (match Native_interp.tensor_of_pt2 (bool_tensor [ 0; 1; 2; 255 ]) with
  | Ok (Tensor.Tensor t as packed) -> (
      Format.printf "%a@." Tensor.pp packed;
      match t.Tensor.payload.Payload.fmt with
      | Payload.Bool ->
          let d = t.Tensor.payload.Payload.data in
          Format.printf "stored bytes: %d,%d,%d,%d@." d.{0} d.{1} d.{2} d.{3}
      | _ -> Format.printf "not a Bool payload@.")
  | Error e ->
      Format.printf "error: %a@." Native_interp.pp_error (Err.Error.kind e));
  [%expect {|
    tensor bool [C=4] {0, 1, 1, 1}
    stored bytes: 0,1,1,1 |}]
