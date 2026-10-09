module Dtype = Pt2_checkpoint_map.Dtype

module Diff = struct
  type t = { actual : string; expected : string; index : int64 list }
end

type verdict =
  | Dtype_differs of { actual : Dtype.t; expected : Dtype.t }
  | Pass
  | Shape_differs of { actual : int64 list; expected : int64 list }
  | Unsupported_dtype of Dtype.t
  | Values_differ

type t = {
  elements : int64;
  first : Diff.t list;
  max_abs_error : float;
  max_rel_error : float;
  mismatches : int64;
  name : string;
  verdict : verdict;
}

let max_reported = 8
let passed t = t.verdict = Pass

(* Round a double to binary32 and back. *)
let r32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* The row-major index of flat element [i] in [shape]. *)
let unravel shape i =
  let dims = Array.of_list shape in
  let rank = Array.length dims in
  let idx = Array.make rank 0L in
  let rest = ref (Int64.of_int i) in
  for d = rank - 1 downto 0 do
    let n = dims.(d) in
    if Int64.compare n 0L > 0 then begin
      idx.(d) <- Int64.rem !rest n;
      rest := Int64.div !rest n
    end
  done;
  Array.to_list idx

let empty name elements verdict =
  {
    elements;
    first = [];
    max_abs_error = 0.;
    max_rel_error = 0.;
    mismatches = 0L;
    name;
    verdict;
  }

let float_close ~single ~atol ~rtol a b =
  if a = b then true
  else if Float.is_nan a || Float.is_nan b then false
  else if Float.abs a = infinity || Float.abs b = infinity then false
  else if single then
    let atol = r32 atol and rtol = r32 rtol in
    let diff = r32 (Float.abs (a -. b)) in
    let tol = r32 (atol +. r32 (rtol *. r32 (Float.abs b))) in
    diff <= tol
  else Float.abs (a -. b) <= atol +. (rtol *. Float.abs b)

let tensor ~atol ~rtol ~name ~(expected : Logical.t) ~(actual : Logical.t) =
  let elements = Logical.numel expected in
  if not (Dtype.equal actual.dtype expected.dtype) then
    empty name elements
      (Dtype_differs { actual = actual.dtype; expected = expected.dtype })
  else if not (List.equal Int64.equal actual.shape expected.shape) then
    empty name elements
      (Shape_differs { actual = actual.shape; expected = expected.shape })
  else
    let n = Int64.to_int elements in
    let mismatches = ref 0L and first = ref [] in
    let max_abs = ref 0. and max_rel = ref 0. in
    let record i ~actual:a ~expected:e =
      mismatches := Int64.succ !mismatches;
      if List.length !first < max_reported then
        first :=
          { Diff.actual = a; expected = e; index = unravel expected.shape i }
          :: !first
    in
    let finish verdict_ok =
      {
        elements;
        first = List.rev !first;
        max_abs_error = !max_abs;
        max_rel_error = !max_rel;
        mismatches = !mismatches;
        name;
        verdict =
          (if verdict_ok && Int64.equal !mismatches 0L then Pass
           else Values_differ);
      }
    in
    match expected.dtype with
    | Dtype.F32 | Dtype.F64 ->
        let single = expected.dtype = Dtype.F32 in
        let render = Printf.sprintf "%.9g" in
        for i = 0 to n - 1 do
          let a = Logical.get_float actual i
          and e = Logical.get_float expected i in
          if not (float_close ~single ~atol ~rtol a e) then
            record i ~actual:(render a) ~expected:(render e);
          if Float.is_finite a && Float.is_finite e then begin
            let d = Float.abs (a -. e) in
            if d > !max_abs then max_abs := d;
            if e <> 0. then begin
              let r = d /. Float.abs e in
              if r > !max_rel then max_rel := r
            end
          end
        done;
        finish true
    | Dtype.BOOL | Dtype.I8 | Dtype.I16 | Dtype.I32 | Dtype.I64 | Dtype.U8 ->
        let render v =
          if expected.dtype = Dtype.BOOL then
            if Int64.equal v 0L then "false" else "true"
          else Int64.to_string v
        in
        for i = 0 to n - 1 do
          let a = Logical.get_int64 actual i
          and e = Logical.get_int64 expected i in
          if not (Int64.equal a e) then
            record i ~actual:(render a) ~expected:(render e)
        done;
        finish true
    | (Dtype.BF16 | Dtype.F16) as d -> empty name elements (Unsupported_dtype d)
