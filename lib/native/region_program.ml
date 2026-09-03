module Local_scope = struct
  type t = { local : Expr.Local_var.t; referenced : Expr.Local_var.t }
end

module Non_invariant = struct
  type t = { local : Expr.Local_var.t; axis : Expr.Axis.t }
end

module Shape_mismatch = struct
  type t = { local : Expr.Local_var.t; declared : Region_local.Shape.t }
end

type error =
  [ `Duplicate_local of Expr.Local_var.t
  | `Expr of Expr.Check.error
  | `Forward_local of Local_scope.t
  | `Local_list_too_large of int
  | `Local_words_over_limit of int
  | `Non_invariant_local of Non_invariant.t
  | `Shape_mismatch of Shape_mismatch.t
  | `Unknown_emitter_local of Expr.Local_var.t
  | `Unknown_local of Local_scope.t ]

type t = {
  partition : Region_partition.t;
  locals : Region_local.t list;
  output : Expr.Value.t;
}

type program = t

let pixel output =
  { partition = Region_partition.singleton; locals = []; output }

let partition t = t.partition
let locals t = t.locals
let output t = t.output
let with_output t output = { t with output }

let pixel_expression t =
  if t.locals = [] && Region_partition.is_singleton t.partition then
    Some t.output
  else None

(* [specialize_pixel]'s own accumulator: what a local resolves to once every
   EARLIER local has already been substituted into it (see [rewrite] below),
   alongside the [size]/[depth] budget its inlined form costs -- shape-
   agnostic, since [preflight] measures a prospective substitution
   structurally, not by whether it fills a [Local] or a [Local_at]. *)
type specialized = {
  binding : Expr.Rewrite.local_binding;
  size : int;
  depth : int;
}

let preflight ~max_size ~max_depth ~locals value =
  match
    Expr.Fold.exceeds_with_locals ~max_size ~max_depth
      ~local:(fun id ->
        let e = Expr.Local_var.Map.find id locals in
        (e.size, e.depth))
      value
  with
  | None ->
      let size, depth =
        Expr.Fold.measure_with_locals ~max_size ~max_depth
          ~local:(fun id ->
            let e = Expr.Local_var.Map.find id locals in
            (e.size, e.depth))
          value
      in
      Err.return (size, depth)
  | Some `Size -> Err.fail (`Expr (`Too_large max_size))
  | Some `Depth -> Err.fail (`Expr (`Too_deep max_depth))

let specialize_pixel ~max_size ~max_depth t =
  match pixel_expression t with
  | Some pixel ->
      let open Err.Syntax in
      let+ () =
        Expr.Check.value ~max_size ~max_depth pixel
        |> Err.map_error (fun e -> `Expr e)
      in
      pixel
  | None ->
      let open Err.Syntax in
      (* Only the SIZE/DEPTH budget is scalar-vs-vector agnostic -- both cost
         the same to inline, since [preflight] measures a prospective
         substitution structurally, not by shape. What differs is HOW each
         local is spliced back in: a scalar local's whole (freshened) value
         replaces a [Local] occurrence outright; a vector local's (freshened)
         body is instead beta-reduced against a [Local_at] occurrence's own
         read index, via [Expr.Rewrite.substitute_locals]'s [Vector] case. *)
      let rewrite locals state value =
        let value, state =
          Expr.Builder.run_from state (Expr.Rewrite.freshen value)
        in
        Expr.Builder.run_from state
          (Expr.Rewrite.substitute_locals
             (fun id ->
               Option.map
                 (fun e -> e.binding)
                 (Expr.Local_var.Map.find_opt id locals))
             value)
      in
      let* locals, state =
        List.fold_left
          (fun acc local ->
            let* locals, state = acc in
            let* size, depth =
              preflight ~max_size ~max_depth ~locals local.Region_local.value
            in
            let value, state = rewrite locals state local.Region_local.value in
            let binding =
              match local.Region_local.shape with
              | Region_local.Shape.Scalar -> Expr.Rewrite.Scalar value
              | Region_local.Shape.Vector { var; _ } ->
                  Expr.Rewrite.Vector { var; body = value }
            in
            Err.return
              ( Expr.Local_var.Map.add local.Region_local.id
                  { binding; size; depth } locals,
                state ))
          (Err.return (Expr.Local_var.Map.empty, Expr.Builder.initial))
          t.locals
      in
      let* () =
        preflight ~max_size ~max_depth ~locals t.output
        |> Err.map (Fun.const ())
      in
      let value, _ = rewrite locals state t.output in
      let+ () =
        Expr.Check.value ~max_size ~max_depth value
        |> Err.map_error (fun e -> `Expr e)
      in
      value

let reconstructs ~max_size ~max_depth ~pixel t =
  let open Err.Syntax in
  let+ specialized = specialize_pixel ~max_size ~max_depth t in
  Expr.Value.equal pixel specialized

let pp_error fmt : [< error ] -> unit = function
  | `Duplicate_local local ->
      Fmt.pf fmt "duplicate local %a" Expr.Local_var.pp local
  | `Expr error -> Expr.Check.pp_error fmt error
  | `Forward_local { Local_scope.local; referenced } ->
      Fmt.pf fmt "local %a refers forward to %a" Expr.Local_var.pp local
        Expr.Local_var.pp referenced
  | `Local_list_too_large limit ->
      Fmt.pf fmt "local list exceeds limit %d" limit
  | `Local_words_over_limit limit ->
      Fmt.pf fmt "total local slot count exceeds limit %d" limit
  | `Non_invariant_local { Non_invariant.local; axis } ->
      Fmt.pf fmt "local %a varies over whole axis %a" Expr.Local_var.pp local
        Expr.Axis.pp axis
  | `Shape_mismatch { Shape_mismatch.local; declared } ->
      Fmt.pf fmt "local %a is read as %s but declared %a" Expr.Local_var.pp
        local
        (match declared with
        | Region_local.Shape.Scalar -> "a vector"
        | Region_local.Shape.Vector _ -> "a scalar")
        Region_local.Shape.pp declared
  | `Unknown_emitter_local local ->
      Fmt.pf fmt "emitter refers to unknown local %a" Expr.Local_var.pp local
  | `Unknown_local { Local_scope.local; referenced } ->
      Fmt.pf fmt "local %a refers to unknown local %a" Expr.Local_var.pp local
        Expr.Local_var.pp referenced

let over_limit limit xs =
  let rec go remaining = function
    | [] -> false
    | _ :: _ when remaining <= 0 -> true
    | _ :: rest -> go (remaining - 1) rest
  in
  go limit xs

let all_local_ids locals =
  List.fold_left
    (fun ids local -> Expr.Local_var.Set.add local.Region_local.id ids)
    Expr.Local_var.Set.empty locals

let all_local_shapes locals =
  List.fold_left
    (fun shapes local ->
      Expr.Local_var.Map.add local.Region_local.id local.Region_local.shape
        shapes)
    Expr.Local_var.Map.empty locals

(* Shape agreement: a [Value.Local] read names a local declared [Scalar], and
   a [Value.Local_at] read names one declared [Vector] -- CLAUDE.md's "closed
   value set is a variant" rule applied to which NODE KIND a local may appear
   as, not just which id. An id absent from [shapes] is a forward/unknown
   reference, already reported by [first_scope_error]/[check_output]; this
   only fires for an id that IS declared, at the wrong kind. *)
let shape_error ~shapes expr =
  let find is_wrong uses =
    Expr.Local_var.Set.to_seq uses
    |> Seq.find_map (fun local ->
        match Expr.Local_var.Map.find_opt local shapes with
        | Some declared when is_wrong declared ->
            Some (`Shape_mismatch { Shape_mismatch.local; declared })
        | Some _ | None -> None)
  in
  let is_vector = function
    | Region_local.Shape.Vector _ -> true
    | Region_local.Shape.Scalar -> false
  in
  let is_scalar = function
    | Region_local.Shape.Scalar -> true
    | Region_local.Shape.Vector _ -> false
  in
  match find is_vector (Expr.Fold.scalar_locals expr) with
  | Some error -> Some error
  | None -> find is_scalar (Expr.Fold.vector_locals expr)

(* Total slot-count across every local, bounds-checked on [Int64] before any
   narrowing: a per-local extent is already bounded (it is one factor of the
   op's own [total_work_bounded]-style product), but the SUM across several
   vector locals is an aggregate, and CLAUDE.md's 32-bit rule is explicit that
   a check on individually-in-range factors does not bound their sum --
   [lib/native] is js_of_ocaml-reachable. [max_size] doubles as the ceiling
   here: it is already the general "how big may this Region program be"
   budget threaded through [create]/[check], and a program's total local
   footprint is part of that same budget rather than a new, separately-tuned
   knob. *)
let checked_slot_total ~limit locals =
  let limit64 = Int64.of_int limit in
  let rec go total = function
    | [] -> Err.return ()
    | local :: rest ->
        let count =
          Int64.of_int (Region_local.Shape.slot_count local.Region_local.shape)
        in
        if Int64.compare total (Int64.sub limit64 count) > 0 then
          Err.fail (`Local_words_over_limit limit)
        else go (Int64.add total count) rest
  in
  go 0L locals

let first_scope_error ~defined ~all ~local expr =
  match
    Expr.Local_var.Set.min_elt_opt
      (Expr.Local_var.Set.diff (Expr.Fold.locals expr) defined)
  with
  | None -> None
  | Some referenced ->
      if Expr.Local_var.Set.mem referenced all then
        Some (`Forward_local { Local_scope.local; referenced })
      else Some (`Unknown_local { Local_scope.local; referenced })

let check ~max_size ~max_depth t =
  let open Err.Syntax in
  if over_limit max_size t.locals then Err.fail (`Local_list_too_large max_size)
  else
    let* () = checked_slot_total ~limit:max_size t.locals in
    let all = all_local_ids t.locals in
    let shapes = all_local_shapes t.locals in
    let rec check_locals remaining defined seen = function
      | [] -> check_output remaining all
      | local :: rest ->
          if Expr.Local_var.Set.mem local.Region_local.id seen then
            Err.fail (`Duplicate_local local.Region_local.id)
          else
            let allowed_free =
              match local.Region_local.shape with
              | Region_local.Shape.Vector { var; _ } ->
                  Expr.Reduce_var.Set.singleton var
              | Region_local.Shape.Scalar -> Expr.Reduce_var.Set.empty
            in
            let* () =
              Expr.Check.fragment ~max_size:remaining ~max_depth ~allowed_free
                ~locals:all local.Region_local.value
              |> Err.map_error (function
                | `Unbound_local referenced ->
                    `Unknown_local
                      { Local_scope.local = local.Region_local.id; referenced }
                | error -> `Expr error)
            in
            let* () =
              match
                first_scope_error ~defined ~all ~local:local.Region_local.id
                  local.Region_local.value
              with
              | None -> Err.return ()
              | Some error -> Err.fail error
            in
            let* () =
              match shape_error ~shapes local.Region_local.value with
              | None -> Err.return ()
              | Some error -> Err.fail error
            in
            let whole = Region_partition.whole_axes t.partition in
            let* () =
              match
                List.find_opt
                  (fun axis ->
                    List.mem axis
                      (Expr.Fold.output_axes local.Region_local.value))
                  whole
              with
              | None -> Err.return ()
              | Some axis ->
                  Err.fail
                    (`Non_invariant_local
                       { Non_invariant.local = local.Region_local.id; axis })
            in
            let consumed = Expr.Fold.size local.Region_local.value in
            check_locals (remaining - consumed)
              (Expr.Local_var.Set.add local.Region_local.id defined)
              (Expr.Local_var.Set.add local.Region_local.id seen)
              rest
    and check_output remaining defined =
      let* () =
        Expr.Check.fragment ~max_size:remaining ~max_depth ~locals:all t.output
        |> Err.map_error (function
          | `Unbound_local local -> `Unknown_emitter_local local
          | error -> `Expr error)
      in
      let* () =
        match
          Expr.Local_var.Set.min_elt_opt
            (Expr.Local_var.Set.diff (Expr.Fold.locals t.output) defined)
        with
        | None -> Err.return ()
        | Some local -> Err.fail (`Unknown_emitter_local local)
      in
      match shape_error ~shapes t.output with
      | None -> Err.return ()
      | Some error -> Err.fail error
    in
    check_locals max_size Expr.Local_var.Set.empty Expr.Local_var.Set.empty
      t.locals

let create ~max_size ~max_depth ~partition ~locals ~output =
  let t = { partition; locals; output } in
  let open Err.Syntax in
  let* () = check ~max_size ~max_depth t in
  Err.return t

module Fold = struct
  let expressions t =
    List.map (fun local -> local.Region_local.value) t.locals @ [ t.output ]

  let sources t =
    List.fold_left
      (fun acc expr -> Expr.Source.Set.union acc (Expr.Fold.sources expr))
      Expr.Source.Set.empty (expressions t)

  let loads t = List.concat_map Expr.Fold.loads (expressions t)

  let intrinsic_sources t =
    List.concat_map Expr.Fold.intrinsic_sources (expressions t)

  let binders t = List.concat_map Expr.Fold.binders (expressions t)

  let intrinsics t =
    List.fold_left
      (fun acc expr -> acc + Expr.Fold.intrinsics expr)
      0 (expressions t)

  let max_depth t =
    List.fold_left
      (fun acc expr -> max acc (Expr.Fold.depth expr))
      0 (expressions t)

  let size t =
    List.fold_left (fun acc expr -> acc + Expr.Fold.size expr) 0 (expressions t)
end

let pp fmt t =
  let names =
    List.mapi (fun i local -> (local.Region_local.id, Fmt.str "l%d" i)) t.locals
    |> List.to_seq |> Expr.Local_var.Map.of_seq
  in
  let local_name id = Expr.Local_var.Map.find_opt id names in
  Fmt.pf fmt "region [%a]" Region_partition.pp t.partition;
  List.iter
    (fun local ->
      Fmt.pf fmt "@\n  let %s : %a = %a"
        (Option.value ~default:"?" (local_name local.Region_local.id))
        Region_local.Shape.pp local.Region_local.shape
        (Expr.Pp.value_open ~names:local_name)
        local.Region_local.value)
    t.locals;
  Fmt.pf fmt "@\n  emit %a" (Expr.Pp.value_open ~names:local_name) t.output

module Builder = struct
  type 'a t =
    Expr.Builder.state -> Region_local.t list -> 'a * Expr.Builder.state

  let run build = fst (build Expr.Builder.initial [])

  let scalar value continue state locals =
    let id, state = Expr.Builder.run_from state Expr.Builder.fresh_local in
    continue (Expr.Value.local id) state
      (Region_local.scalar ~id ~value :: locals)

  (* [body] receives its own per-element index, symbolically -- exactly the
     shape [Expr.Builder.reduction]'s own body callback has -- and [continue]
     receives a READER, not a value: a vector local has no single value to
     hand back, only [Expr.Value.local_at id] applied at whatever index the
     caller supplies (an enclosing reduction's own bound variable, typically).
     [var] is minted the same trusted way [Expr.Builder.reduction] mints its
     own binder ([Expr.Builder.fresh_reduce], already public); it occurs FREE
     in [value]'s result, which is what lets [specialize_pixel] beta-reduce it
     back at each read site. *)
  let vector ~extent value continue state locals =
    let id, state = Expr.Builder.run_from state Expr.Builder.fresh_local in
    let var, state = Expr.Builder.run_from state Expr.Builder.fresh_reduce in
    let body, state =
      Expr.Builder.run_from state (value (Expr.Index.reduce var))
    in
    continue
      (fun idx -> Expr.Value.local_at id idx)
      state
      (Region_local.vector ~id ~var ~extent ~value:body :: locals)

  let finish ~max_size ~max_depth ~partition ~output state locals =
    ( create ~max_size ~max_depth ~partition ~locals:(List.rev locals) ~output,
      state )
end
