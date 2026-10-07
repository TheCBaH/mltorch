(* The comparison protocol between execution routes. Two routes agree only
   when both succeed with the same outputs and events, or both fail with the
   same row (kind, decoded static identity, payload and invocation) after the
   same events. A defect, unsupported capability or exhausted fuel on either
   side is never agreement — "both failed" is not a verdict. Integers and
   signed zeros compare exactly; a NaN matches any NaN (payload and sign are
   not portable). Raw site numbers compare only under an explicitly shared
   numbering convention. *)

module Sites = struct
  type t =
    | Decoded  (** compare decoded identity; raw site numbers may differ *)
    | Normalized  (** both routes number sites by one convention *)
end

module Difference = struct
  type t =
    | Event of { event : Mir_event.t; expected : int64; actual : int64 }
    | Inconclusive of Mir_observation.Status.t * Mir_observation.Status.t
    | Missing_output of Expr.Source.t
    | Output of {
        source : Expr.Source.t;
        index : int;
        expected : Mir_const.t option;
        actual : Mir_const.t option;
      }
    | Row of Mir_observation.Row.t * Mir_observation.Row.t
    | Status of Mir_observation.Status.t * Mir_observation.Status.t

  let pp_status fmt = function
    | Mir_observation.Status.Defect d ->
        Fmt.pf fmt "defect(%s)" (Mir_observation.Defect.name d)
    | Mir_observation.Status.Failure r ->
        Fmt.pf fmt "failure(%a)" Mir_failure.pp r.Mir_observation.Row.failure
    | Mir_observation.Status.Fuel_exhausted -> Fmt.string fmt "fuel"
    | Mir_observation.Status.Success -> Fmt.string fmt "success"
    | Mir_observation.Status.Unsupported s -> Fmt.pf fmt "unsupported(%s)" s

  let pp_cell fmt = function
    | Some c -> Mir_const.pp fmt c
    | None -> Fmt.string fmt "undefined"

  let pp_row fmt (r : Mir_observation.Row.t) =
    Fmt.pf fmt "%a(%a)%a%a" Mir_failure.pp r.Mir_observation.Row.failure
      Fmt.(list ~sep:(any ", ") Mir_const.pp)
      r.Mir_observation.Row.payload
      Fmt.(option (any " invocation " ++ int32))
      r.Mir_observation.Row.invocation
      Fmt.(option (any " " ++ Mir_id.Site.pp))
      r.Mir_observation.Row.site

  let pp fmt = function
    | Event { event; expected; actual } ->
        Fmt.pf fmt "event %s: %Ld vs %Ld" (Mir_event.name event) expected actual
    | Inconclusive (a, b) ->
        Fmt.pf fmt "inconclusive: %a vs %a" pp_status a pp_status b
    | Missing_output s -> Fmt.pf fmt "output %a missing" Expr.Source.pp s
    | Output { source; index; expected; actual } ->
        Fmt.pf fmt "output %a[%d]: %a vs %a" Expr.Source.pp source index pp_cell
          expected pp_cell actual
    | Row (a, b) -> Fmt.pf fmt "failure row: %a vs %a" pp_row a pp_row b
    | Status (a, b) -> Fmt.pf fmt "status: %a vs %a" pp_status a pp_status b
end

(* Exact bits, but one class for every NaN. *)
let cell_equal (a : Mir_const.t) (b : Mir_const.t) =
  Mir_type.equal a.Mir_const.ty b.Mir_const.ty
  &&
  match a.Mir_const.ty with
  | Mir_type.F64 ->
      Core.Float_bits.equal_portable
        (Int64.float_of_bits a.Mir_const.bits)
        (Int64.float_of_bits b.Mir_const.bits)
  | Mir_type.F32 ->
      let f x = Int32.float_of_bits (Int64.to_int32 x) in
      Core.Float_bits.equal_portable (f a.Mir_const.bits) (f b.Mir_const.bits)
  | Mir_type.Flags | Mir_type.Int _ | Mir_type.Mask _ | Mir_type.Order
  | Mir_type.Pred | Mir_type.Ptr | Mir_type.Vec _ ->
      Int64.equal a.Mir_const.bits b.Mir_const.bits

let row_equal ~sites (a : Mir_observation.Row.t) (b : Mir_observation.Row.t) =
  Mir_failure.equal a.Mir_observation.Row.failure b.Mir_observation.Row.failure
  && List.equal cell_equal a.Mir_observation.Row.payload
       b.Mir_observation.Row.payload
  && Option.equal Int32.equal a.Mir_observation.Row.invocation
       b.Mir_observation.Row.invocation
  &&
  match sites with
  | Sites.Decoded -> true
  | Sites.Normalized ->
      Option.equal Mir_id.Site.equal a.Mir_observation.Row.site
        b.Mir_observation.Row.site

let first_some f l = List.find_map f l

let events ~(expected : Mir_observation.t) ~(actual : Mir_observation.t) =
  first_some
    (fun e ->
      let count (o : Mir_observation.t) =
        Option.value ~default:0L (List.assoc_opt e o.Mir_observation.events)
      in
      let x = count expected and y = count actual in
      if Int64.equal x y then None
      else Some (Difference.Event { event = e; expected = x; actual = y }))
    Mir_event.all

(* The first output cell [cell source index expected actual] rejects. *)
let outputs ~cell ~(expected : Mir_observation.t) ~(actual : Mir_observation.t)
    =
  first_some
    (fun (o : Mir_observation.Output.t) ->
      let source = o.Mir_observation.Output.source in
      match
        List.find_opt
          (fun (p : Mir_observation.Output.t) ->
            Expr.Source.equal p.Mir_observation.Output.source source)
          actual.Mir_observation.outputs
      with
      | None -> Some (Difference.Missing_output source)
      | Some p ->
          let a = o.Mir_observation.Output.cells
          and b = p.Mir_observation.Output.cells in
          let n = max (Array.length a) (Array.length b) in
          let get c i = if i < Array.length c then c.(i) else None in
          let rec go i =
            if i >= n then None
            else if cell source i (get a i) (get b i) then go (i + 1)
            else
              Some
                (Difference.Output
                   { source; index = i; expected = get a i; actual = get b i })
          in
          go 0)
    expected.Mir_observation.outputs

let defined_equal a b =
  match (a, b) with
  | Some a, Some b -> cell_equal a b
  | None, None -> true
  | Some _, None | None, Some _ -> false

let decisive = function
  | Mir_observation.Status.Success | Mir_observation.Status.Failure _ -> true
  | Mir_observation.Status.Defect _ | Mir_observation.Status.Fuel_exhausted
  | Mir_observation.Status.Unsupported _ ->
      false

let with_cells ~sites ~cell ~(expected : Mir_observation.t)
    ~(actual : Mir_observation.t) =
  let s = expected.Mir_observation.status
  and t = actual.Mir_observation.status in
  if not (decisive s && decisive t) then Error (Difference.Inconclusive (s, t))
  else
    match (s, t) with
    | Mir_observation.Status.Success, Mir_observation.Status.Success -> (
        match outputs ~cell ~expected ~actual with
        | Some d -> Error d
        | None -> (
            match events ~expected ~actual with
            | Some d -> Error d
            | None -> Ok ()))
    | Mir_observation.Status.Failure a, Mir_observation.Status.Failure b -> (
        if not (row_equal ~sites a b) then Error (Difference.Row (a, b))
        else
          (* the failure prefix: events before the failure are observable *)
          match events ~expected ~actual with
          | Some d -> Error d
          | None -> Ok ())
    | _ -> Error (Difference.Status (s, t))

let observations ?(sites = Sites.Decoded) ~expected ~actual () =
  with_cells ~sites ~cell:(fun _ _ -> defined_equal) ~expected ~actual

let cell_of (o : Mir_observation.t) source i =
  Option.bind
    (List.find_opt
       (fun (p : Mir_observation.Output.t) ->
         Expr.Source.equal p.Mir_observation.Output.source source)
       o.Mir_observation.outputs)
    (fun p ->
      let c = p.Mir_observation.Output.cells in
      if i < Array.length c then c.(i) else None)

(* An external engine whose validated planning summary permits a relaxed
   multiply-add may produce, cell by cell, either the fused or the unfused
   oracle's value. This boundary alone accepts either: a Machine IR route is
   always compared strictly with the fused oracle through [observations]. *)
let relaxed_external ~(planning : Mir_planning.t) ~(fused : Mir_observation.t)
    ~(unfused : Mir_observation.t) ~(external_ : Mir_observation.t) =
  match planning.Mir_planning.fma with
  | Mir_planning.Fma.Exact | Mir_planning.Fma.Forbidden ->
      observations ~expected:fused ~actual:external_ ()
  | Mir_planning.Fma.Relaxed_madd ->
      let cell source i a b =
        defined_equal a b || defined_equal (cell_of unfused source i) b
      in
      with_cells ~sites:Sites.Decoded ~cell ~expected:fused ~actual:external_
