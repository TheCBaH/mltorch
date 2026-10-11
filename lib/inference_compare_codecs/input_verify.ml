open Inference_compare_schema

type difference =
  | Spec_differs of { expected : Tensor_spec.t; actual : Tensor_spec.t }
  | Element_differs of { index : int64 }

type outcome =
  | Equal of Recipe_eval.Lineage.t
  | Different of { difference : difference; lineage : Recipe_eval.Lineage.t }

let ( let* ) = Err.bind

let first_difference ctx ~expected ~actual =
  let es = Logical_value.spec expected and as_ = Logical_value.spec actual in
  if not (Tensor_spec.equal es as_) then
    Ok (Some (Spec_differs { expected = es; actual = as_ }))
  else
    let a = Logical_value.contents expected
    and b = Logical_value.contents actual in
    if String.equal a b then Ok None
    else
      let width = (Logical_value.codec ctx expected).byte_width in
      (* Same spec means the same byte length; scan by element so the index is
         an element position, not a byte offset. *)
      let n = String.length a / width in
      let rec go i =
        if i >= n then None
        else
          let rec same k =
            k >= width
            || (a.[(i * width) + k] = b.[(i * width) + k] && same (k + 1))
          in
          if same 0 then go (i + 1) else Some i
      in
      match go 0 with
      | Some i -> Ok (Some (Element_differs { index = Int64.of_int i }))
      | None -> Ok None

let verify ctx env checked ~imported ~expected =
  let inputs =
    { Recipe_eval.canonical = (fun _ -> None); boundary = imported }
  in
  let* reconstructed, lineage = Recipe_eval.evaluate ctx env checked inputs in
  let result = first_difference ctx ~expected ~actual:reconstructed in
  Logical_value.drop reconstructed;
  match result with
  | Error e -> Error e
  | Ok None -> Ok (Equal lineage)
  | Ok (Some difference) -> Ok (Different { difference; lineage })
