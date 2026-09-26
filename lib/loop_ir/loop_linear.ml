(* An index as an exact integer linear combination of atoms plus a constant:
   [Add], [Scale] and [Const] are unfolded, and every other node (a variable,
   a temporary, a division, a [Min]/[Max]/[Clamp_low]) is an opaque atom,
   compared structurally.

   This is the mathematical value only: it says nothing about the overflow of
   the [Add]/[Scale] nodes it unfolds, so a caller that rebuilds an index from
   a form must prove the original and the result in the domain itself
   ([Loop_range.proven]). Every coefficient and the constant are bounded to
   [Loop_range.domain] in [int64] before they are narrowed to [int], so a form
   is exact under js_of_ocaml's 32-bit [int] too; a combination that leaves it
   is [None]. *)

type t = { terms : (Loop_index.t * int) list; const : int }
(** [terms] in order of first appearance, no zero coefficient, no atom twice. *)

let narrow x =
  if Loop_range.within ~inner:(Loop_range.point x) ~outer:Loop_range.domain then
    Some (Int64.to_int x)
  else None

let ( let* ) = Option.bind
let const n = { terms = []; const = n }
let atom a = { terms = [ (a, 1) ]; const = 0 }

let add f g =
  let* const =
    narrow
      (Loop_range.saturating_add (Int64.of_int f.const) (Int64.of_int g.const))
  in
  let* terms =
    List.fold_left
      (fun acc (a, c) ->
        let* acc = acc in
        match List.assoc_opt a acc with
        | None -> Some (acc @ [ (a, c) ])
        | Some c' ->
            let* sum =
              narrow
                (Loop_range.saturating_add (Int64.of_int c) (Int64.of_int c'))
            in
            Some
              (List.filter_map
                 (fun (a', c'') ->
                   if Stdlib.( = ) a' a then
                     if sum = 0 then None else Some (a', sum)
                   else Some (a', c''))
                 acc))
      (Some f.terms) g.terms
  in
  Some { terms; const }

let scale k f =
  if k = 0 then Some (const 0)
  else
    let mul c = narrow (Loop_range.saturating_mul k (Int64.of_int c)) in
    let* const = mul f.const in
    let* terms =
      List.fold_left
        (fun acc (a, c) ->
          let* acc = acc in
          let* c = mul c in
          Some ((a, c) :: acc))
        (Some []) f.terms
    in
    Some { terms = List.rev terms; const }

let sub f g =
  let* g = scale (-1) g in
  add f g

let rec of_index : Loop_index.t -> t option = function
  | Loop_index.Add (a, b) ->
      let* a = of_index a in
      let* b = of_index b in
      add a b
  | Loop_index.Const n -> Some (const n)
  | Loop_index.Scale (k, a) ->
      let* a = of_index a in
      scale k a
  | ( Loop_index.Ceil_div_pos _ | Loop_index.Clamp_low _
    | Loop_index.Floor_div_pos _ | Loop_index.Max _ | Loop_index.Min _
    | Loop_index.Temp _ | Loop_index.Var _ ) as a ->
      Some (atom a)

(* Positive terms first, so a backend that prints [a + -k * b] as
   [a - k * b] reads naturally; with none, the constant leads
   ([9 + -1 * i] rather than [-1 * i + 9]). *)
let to_index f : Loop_index.t =
  let term (a, c) = if c = 1 then a else Loop_index.Scale (c, a) in
  let pos, neg = List.partition (fun (_, c) -> c > 0) f.terms in
  let leading, rest =
    match (pos, f.const) with
    | [], 0 -> ([], neg)
    | [], c -> ([ Loop_index.Const c ], neg)
    | _, 0 -> (List.map term pos, neg)
    | _, c -> (List.map term pos @ [ Loop_index.Const c ], neg)
  in
  match leading @ List.map term rest with
  | [] -> Loop_index.Const 0
  | first :: more ->
      List.fold_left (fun acc t -> Loop_index.Add (acc, t)) first more

let coefficient f a = Option.value ~default:0 (List.assoc_opt a f.terms)

(* [m * (x - d * floor (x / d))] is in [m * [0, d - 1]] whatever [x] is: the
   relational fact an interval over [x] and [floor (x / d)] separately cannot
   see, and the whole of an unflattened reshape's bounds check. A form holding
   [floor (x / d)] at coefficient [-m * d] is split into that remainder and
   what is left, and both enclosures are intersected with the plain one. The
   depth bounds the (small) number of nested divisions a form can peel. *)
let rec range ?(depth = 4) env f : Loop_range.t =
  let plain =
    List.fold_left
      (fun (acc : Loop_range.t) (a, c) ->
        let r = Loop_range.scale c (Loop_range.of_index env a) in
        {
          Loop_range.lo = Loop_range.saturating_add acc.lo r.lo;
          hi = Loop_range.saturating_add acc.hi r.hi;
        })
      (Loop_range.point (Int64.of_int f.const))
      f.terms
  in
  if depth = 0 then plain
  else
    List.fold_left
      (fun (best : Loop_range.t) (a, c) ->
        match a with
        | Loop_index.Floor_div_pos (x, d) when c mod d = 0 -> (
            let m = -(c / d) in
            let split =
              let* x = of_index x in
              let* d_floor = scale d (atom a) in
              let* remainder = sub x d_floor in
              let* scaled = scale m remainder in
              sub f scaled
            in
            match split with
            | None -> best
            | Some rest ->
                let r = range ~depth:(depth - 1) env rest in
                let rem =
                  Loop_range.scale m
                    { Loop_range.lo = 0L; hi = Int64.of_int (d - 1) }
                in
                {
                  Loop_range.lo =
                    Stdlib.max best.lo (Loop_range.saturating_add r.lo rem.lo);
                  hi =
                    Stdlib.min best.hi (Loop_range.saturating_add r.hi rem.hi);
                })
        | _ -> best)
      plain f.terms
