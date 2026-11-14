type t = { policy : string; values : float array }

type mismatch =
  | Length_differs of { expected : Count.t; actual : Count.t }
  | Policy_differs of { expected : string; actual : string }
  | Value_differs of { index : Count.t; expected : int64; actual : int64 }

let compare g ~policy actual =
  if not (String.equal g.policy policy) then
    Error (Policy_differs { expected = g.policy; actual = policy })
  else if Array.length g.values <> Array.length actual then
    Error
      (Length_differs
         {
           expected = Count.of_int (Array.length g.values);
           actual = Count.of_int (Array.length actual);
         })
  else
    let rec first i =
      if i >= Array.length actual then Ok ()
      else
        let e = Int64.bits_of_float g.values.(i)
        and a = Int64.bits_of_float actual.(i) in
        if Int64.equal e a then first (i + 1)
        else
          Error
            (Value_differs { index = Count.of_int i; expected = e; actual = a })
    in
    first 0

let with_defect values =
  let v = Array.copy values in
  if Array.length v > 0 then
    v.(0) <- Int64.float_of_bits (Int64.logxor (Int64.bits_of_float v.(0)) 1L);
  v

let pp_mismatch ppf = function
  | Length_differs { expected; actual } ->
      Fmt.pf ppf "length %a, expected %a" Count.pp actual Count.pp expected
  | Policy_differs { expected; actual } ->
      Fmt.pf ppf "compared under %s, the golden was made under %s" actual
        expected
  | Value_differs { index; expected; actual } ->
      Fmt.pf ppf "element %a: %016Lx, expected %016Lx" Count.pp index actual
        expected
