type t = {
  cells : int;
  failing : int;
  nonfinite_actual : int;
  nonfinite_reference : int;
  nonfinite_mismatch : int;
  max_abs : float;
  max_rel : float;
  max_normalized : float;
  ulp_buckets : (string * int) list;
}

let bucket_names = [ "0"; "1"; "2-4"; "5-16"; "17-256"; ">256" ]

(* Binary32 values on one integer line, so adjacent floats differ by one and a
   crossing of zero is continuous. *)
let ordered x =
  let b = Int32.to_int (Int32.bits_of_float x) in
  if b < 0 then -(b land 0x7FFFFFFF) else b

let bucket d =
  if d = 0 then 0
  else if d = 1 then 1
  else if d <= 4 then 2
  else if d <= 16 then 3
  else if d <= 256 then 4
  else 5

let kind x =
  if Float.is_nan x then `Nan
  else if x = Float.infinity then `Pos_inf
  else if x = Float.neg_infinity then `Neg_inf
  else `Finite

let compare ~atol ~rtol ~actual ~reference =
  let (Tensor.Tensor ta) = actual in
  let failing = ref 0 and na = ref 0 and nr = ref 0 and nm = ref 0 in
  let max_abs = ref 0. and max_rel = ref 0. and scale = ref 0. in
  let buckets = Array.make (List.length bucket_names) 0 in
  let cells = ref 0 in
  Vec6.iter ta.Tensor.shape (fun c ->
      incr cells;
      let a = Tensor.read_at actual (Vec6.get c)
      and r = Tensor.read_at reference (Vec6.get c) in
      let ka = kind a and kr = kind r in
      if ka <> `Finite then incr na;
      if kr <> `Finite then incr nr;
      if ka <> `Finite || kr <> `Finite then (
        if ka <> kr then (
          incr nm;
          incr failing))
      else
        let d = Float.abs (a -. r) in
        if d > atol +. (rtol *. Float.abs r) then incr failing;
        max_abs := Float.max !max_abs d;
        if r <> 0. then max_rel := Float.max !max_rel (d /. Float.abs r);
        scale := Float.max !scale (Float.abs r);
        let u = bucket (abs (ordered a - ordered r)) in
        buckets.(u) <- buckets.(u) + 1);
  {
    cells = !cells;
    failing = !failing;
    nonfinite_actual = !na;
    nonfinite_reference = !nr;
    nonfinite_mismatch = !nm;
    max_abs = !max_abs;
    max_rel = !max_rel;
    max_normalized = (if !scale = 0. then !max_abs else !max_abs /. !scale);
    ulp_buckets = List.mapi (fun i n -> (n, buckets.(i))) bucket_names;
  }

let outputs ~atol ~rtol ~reference actual =
  List.map
    (fun (id, t) ->
      (id, compare ~atol ~rtol ~actual:t ~reference:(reference id)))
    actual

let merge a b =
  {
    cells = a.cells + b.cells;
    failing = a.failing + b.failing;
    nonfinite_actual = a.nonfinite_actual + b.nonfinite_actual;
    nonfinite_reference = a.nonfinite_reference + b.nonfinite_reference;
    nonfinite_mismatch = a.nonfinite_mismatch + b.nonfinite_mismatch;
    max_abs = Float.max a.max_abs b.max_abs;
    max_rel = Float.max a.max_rel b.max_rel;
    max_normalized = Float.max a.max_normalized b.max_normalized;
    ulp_buckets =
      List.map2 (fun (n, x) (_, y) -> (n, x + y)) a.ulp_buckets b.ulp_buckets;
  }

let pp ppf t =
  Fmt.pf ppf
    "%d cells, %d outside tolerance; max abs %.3g, max rel %.3g, max \
     normalized %.3g; nonfinite %d actual / %d reference / %d mismatched; ulp \
     %a"
    t.cells t.failing t.max_abs t.max_rel t.max_normalized t.nonfinite_actual
    t.nonfinite_reference t.nonfinite_mismatch
    Fmt.(list ~sep:(any " ") (pair ~sep:(any ":") string int))
    t.ulp_buckets

let total = function
  | [] -> None
  | (_, d) :: rest -> Some (List.fold_left (fun a (_, d) -> merge a d) d rest)
