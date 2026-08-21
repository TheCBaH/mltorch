open Bigarray

let pp_ints xs = String.concat "; " (Array.to_list (Array.map string_of_int xs))

let int64_tensor ~sizes ~strides ~storage_offset values =
  let data = Bytes.create (8 * Array.length values) in
  Array.iteri (fun i value -> Bytes.set_int64_le data (8 * i) value) values;
  Pt2_tensor.{ dtype = Pt2_dtype.Int64; sizes; strides; storage_offset; data }

let%expect_test "int64 bridge preserves signed strided storage" =
  let source =
    int64_tensor ~sizes:[ 2; 2 ] ~strides:[ 1; 2 ] ~storage_offset:1
      [| 99L; -2L; Int64.max_int; 7L; Int64.min_int |]
  in
  let tensor = Pt2_aten.to_tensor source in
  let values =
    Aten_tensor.materialize_for_raw_read tensor
    |> Aten_tensor.as_int64 |> Option.get
  in
  Printf.printf "dtype=int64 shape=[%s] strides=[%s] offset=%d values=[%s]\n"
    (pp_ints (Aten_tensor.shape tensor))
    (pp_ints (Aten_tensor.strides tensor))
    (Aten_tensor.storage_offset tensor)
    (String.concat "; "
       (List.init (Array1.dim values) (fun i -> Int64.to_string values.{i})));
  [%expect
    {|dtype=int64 shape=[2; 2] strides=[1; 2] offset=0 values=[-2; 7; 9223372036854775807; -9223372036854775808]
|}]
