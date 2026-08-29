open Loop_ir_test
open Loop_fixtures

(* The input the Region and scan fixtures read: two rows of three. *)

let data = [| 1.; 2.; 3.; 10.; 20.; 30. |]

let bind id =
  if Tensor_id.equal id (tid 0) then
    Some
      (f32_tensor Loop_programs.rows_shape (fun c ->
           data.((Vec6.offset Loop_programs.rows_shape c :> int))))
  else None
