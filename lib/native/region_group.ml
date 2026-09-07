module Emitter = struct
  type t = {
    output_shape : Vec6.shape;
    partition : Region_partition.t;
    key_axes : (Expr.Axis.t * Expr.Axis.t) list;
    output : Expr.Value.t;
  }
end

module Extent_mismatch = struct
  type t = { canonical : Expr.Axis.t; physical : Expr.Axis.t }
end

type mapping_error =
  | Duplicate_canonical_axis of Expr.Axis.t
  | Duplicate_target_axis of Expr.Axis.t
  | Extent_mismatch of Extent_mismatch.t
  | Target_not_singleton of Expr.Axis.t
  | Uncovered_canonical_axis of Expr.Axis.t
  | Uncovered_physical_axis of Expr.Axis.t

type error =
  [ `Empty_emitters
  | `Mapping of int * mapping_error
  | `Program of Region_program.error
  | `Scan of Expr.Scan.error
  | `Unknown_emitter of int ]

type t = {
  canonical_shape : Vec6.shape;
  canonical_partition : Region_partition.t;
  locals : Region_local.t list;
  emitters : Emitter.t list;
}

let canonical_shape t = t.canonical_shape
let canonical_partition t = t.canonical_partition
let locals t = t.locals
let emitters t = t.emitters
let emitter t i = List.nth_opt t.emitters i

(* The raw (unprojected) expressions ordinal [i]'s shared locals and own
   emitter together consist of -- every [Expr.Fold] query used below (and by
   [Ref]'s callers) is invariant under [project]'s substitution:
   [substitute_output] (see [coord_of_key_axes] below) replaces one [Output]
   leaf with another, touching neither sources, binders/reducer identities,
   intrinsic descriptors, nor tree shape, so each of these can be computed
   directly on the raw group, with no [project] (and so no [Err] channel)
   needed. *)
let expressions t i =
  match emitter t i with
  | None -> None
  | Some e ->
      Some
        (List.map
           (fun (l : Region_local.t) ->
             Region_local.Rhs.value l.Region_local.rhs)
           t.locals
        @ [ e.Emitter.output ])

let fold_expressions ~zero ~f t i =
  Option.map (List.fold_left f zero) (expressions t i)

let sources t i =
  fold_expressions ~zero:Expr.Source.Set.empty
    ~f:(fun acc e -> Expr.Source.Set.union acc (Expr.Fold.sources e))
    t i

let max_depth t i =
  fold_expressions ~zero:0
    ~f:(fun acc e -> Stdlib.max acc (Expr.Fold.depth e))
    t i

let intrinsic_sources t i =
  fold_expressions ~zero:[]
    ~f:(fun acc e -> List.rev_append (Expr.Fold.intrinsic_sources e) acc)
    t i

let binders t i =
  fold_expressions ~zero:[]
    ~f:(fun acc e -> List.rev_append (Expr.Fold.binders e) acc)
    t i

let intrinsics t i =
  fold_expressions ~zero:0 ~f:(fun acc e -> acc + Expr.Fold.intrinsics e) t i

let pp_mapping_error fmt = function
  | Duplicate_canonical_axis a ->
      Fmt.pf fmt "duplicate canonical axis %a" Expr.Axis.pp a
  | Duplicate_target_axis a ->
      Fmt.pf fmt "duplicate target axis %a" Expr.Axis.pp a
  | Extent_mismatch { Extent_mismatch.canonical; physical } ->
      Fmt.pf fmt "canonical axis %a and physical axis %a do not agree in extent"
        Expr.Axis.pp canonical Expr.Axis.pp physical
  | Target_not_singleton a ->
      Fmt.pf fmt "target axis %a is not singleton in the emitter's partition"
        Expr.Axis.pp a
  | Uncovered_canonical_axis a ->
      Fmt.pf fmt "canonical key axis %a is not covered by any key mapping"
        Expr.Axis.pp a
  | Uncovered_physical_axis a ->
      Fmt.pf fmt "physical singleton axis %a is not covered by any key mapping"
        Expr.Axis.pp a

let pp_error fmt : [< error ] -> unit = function
  | `Empty_emitters ->
      Fmt.string fmt "a region group needs at least one emitter"
  | `Mapping (i, e) -> Fmt.pf fmt "emitter %d: %a" i pp_mapping_error e
  | `Program error -> Region_program.pp_error fmt error
  | `Scan error -> Expr.Scan.pp_error fmt error
  | `Unknown_emitter i -> Fmt.pf fmt "unknown emitter ordinal %d" i

(* The canonical key axes are whichever axes some emitter actually DECLARES
   as the canonical side of its [key_axes] mapping -- never derived from
   [canonical_shape]'s own extents. A declared batch axis remains the key
   axis even at extent 1 (design record §3.2's "a declared mapped batch axis
   remains valid when its extent is one" -- batch=1 is a real, exercised
   LSTM configuration): deriving key-ness from "extent > 1" instead would
   make a batch-1 shared local's own [Expr.Index.output] read of that axis
   look, incorrectly, like it varies over a Whole axis. *)
let declared_canonical_axes emitters =
  List.sort_uniq compare
    (List.concat_map
       (fun (e : Emitter.t) -> List.map fst e.Emitter.key_axes)
       emitters)

(* [canonical_partition]'s own Whole axes are exactly the NON-key axes, so
   [Region_program.check]'s existing [Non_invariant_local] rule -- "a local
   may not vary over a Whole axis" -- becomes, for FREE, "a shared local may
   only read the declared canonical key axes via [Expr.Index.output]" (see
   [check_shared_locals]). *)
let canonical_partition_of declared_axes =
  let whole =
    List.filter (fun a -> not (List.mem a declared_axes)) Expr.Axis.all
  in
  match Region_partition.of_whole_axes whole with
  | Ok p -> p
  | Error _ ->
      (* [whole] is built by filtering [Expr.Axis.all] once, so it cannot
         contain a duplicate -- the only way [of_whole_axes] fails. *)
      assert false

let find_duplicate axes =
  let rec go seen = function
    | [] -> None
    | a :: rest -> if List.mem a seen then Some a else go (a :: seen) rest
  in
  go [] axes

(* Checked bijection between the canonical key coordinate and one emitter's
   own physical Singleton coordinate -- design record §3.2, points 1-2. *)
let validate_mapping ~canonical_shape (e : Emitter.t) =
  let open Err.Syntax in
  let canonicals = List.map fst e.Emitter.key_axes in
  let targets = List.map snd e.Emitter.key_axes in
  match find_duplicate canonicals with
  | Some a -> Err.fail (Duplicate_canonical_axis a)
  | None -> (
      match find_duplicate targets with
      | Some a -> Err.fail (Duplicate_target_axis a)
      | None ->
          let* () =
            Err.List.iter
              (fun (c, p) ->
                let* () =
                  match Region_partition.mode e.Emitter.partition p with
                  | Region_partition.Axis_mode.Singleton -> Err.return ()
                  | Region_partition.Axis_mode.Whole ->
                      Err.fail (Target_not_singleton p)
                in
                let c_extent = Dim.to_int (Vec6.get canonical_shape c) in
                let p_extent = Dim.to_int (Vec6.get e.Emitter.output_shape p) in
                if c_extent = p_extent then Err.return ()
                else
                  Err.fail
                    (Extent_mismatch
                       { Extent_mismatch.canonical = c; physical = p }))
              e.Emitter.key_axes
          in
          let canonical_key_axes =
            List.filter
              (fun a -> Dim.to_int (Vec6.get canonical_shape a) > 1)
              Expr.Axis.all
          in
          let* () =
            Err.List.iter
              (fun a ->
                if List.mem a canonicals then Err.return ()
                else Err.fail (Uncovered_canonical_axis a))
              canonical_key_axes
          in
          let physical_key_axes =
            List.filter
              (fun a ->
                (match Region_partition.mode e.Emitter.partition a with
                  | Region_partition.Axis_mode.Singleton -> true
                  | Region_partition.Axis_mode.Whole -> false)
                && Dim.to_int (Vec6.get e.Emitter.output_shape a) > 1)
              Expr.Axis.all
          in
          Err.List.iter
            (fun a ->
              if List.mem a targets then Err.return ()
              else Err.fail (Uncovered_physical_axis a))
            physical_key_axes)

let check_shared_locals ~max_size ~max_depth ~canonical_partition locals =
  let open Err.Syntax in
  let+ (_ : Region_program.t) =
    Err.map_error
      (fun e -> `Program e)
      (Region_program.create ~max_size ~max_depth ~partition:canonical_partition
         ~locals ~output:(Expr.Value.const 0.))
  in
  ()

let coord_of_key_axes key_axes =
  List.fold_left
    (fun c (canon, phys) -> Expr.Coord.set c canon (Expr.Index.output phys))
    (Expr.Coord.of_fn (fun a -> Expr.Index.output a))
    key_axes

(* Rewrites one shared local's output-axis references from canonical to one
   emitter's physical axes, per design record §3.3: reducer/local identities
   are never touched, only [Expr.Index.Output] leaves. A scan local is
   projected through the same closed placeholder wrapper [Region_local.Rhs]
   itself uses to make a descriptor foldable (see [Region_local.Rhs.value]) --
   [Expr.Rewrite.rebuild]'s [Scan_at] case always rebuilds a [Scan_at], so
   unwrapping it back is safe by construction, not merely by convention. *)
let project_local coord (local : Region_local.t) =
  let id = local.Region_local.id in
  match local.Region_local.rhs with
  | Region_local.Rhs.Scalar v ->
      Region_local.scalar ~id ~value:(Expr.Rewrite.substitute_output coord v)
  | Region_local.Rhs.Vector { extent; var; body } ->
      Region_local.vector ~id ~var ~extent
        ~value:(Expr.Rewrite.substitute_output coord body)
  | Region_local.Rhs.Scan s -> (
      let wrapped =
        Expr.Value.scan_at s ~row:Expr.Index.zero ~lane:Expr.Index.zero
      in
      match Expr.Rewrite.substitute_output coord wrapped with
      | Expr.Value.Scan_at (s', _, _) -> Region_local.scan ~id ~scan:s'
      | _ -> assert false)

let project_raw ~max_size ~max_depth ~locals (e : Emitter.t) :
    (Region_program.t, error) Err.t =
  let coord = coord_of_key_axes e.Emitter.key_axes in
  let projected_locals = List.map (project_local coord) locals in
  Err.map_error
    (fun err -> `Program err)
    (Region_program.create ~max_size ~max_depth ~partition:e.Emitter.partition
       ~locals:projected_locals ~output:e.Emitter.output)

let project ~max_size ~max_depth t ordinal =
  match List.nth_opt t.emitters ordinal with
  | None -> Err.fail (`Unknown_emitter ordinal)
  | Some e -> project_raw ~max_size ~max_depth ~locals:t.locals e

let create ~max_size ~max_depth ~canonical_shape ~locals ~emitters =
  match emitters with
  | [] -> Err.fail `Empty_emitters
  | _ :: _ ->
      let open Err.Syntax in
      let canonical_partition =
        canonical_partition_of (declared_canonical_axes emitters)
      in
      let* () =
        Err.List.iter
          (fun (i, e) ->
            Err.map_error
              (fun err -> `Mapping (i, err))
              (validate_mapping ~canonical_shape e))
          (List.mapi (fun i e -> (i, e)) emitters)
      in
      let* () =
        check_shared_locals ~max_size ~max_depth ~canonical_partition locals
      in
      let* () =
        Err.List.iter
          (fun e ->
            let+ (_ : Region_program.t) =
              project_raw ~max_size ~max_depth ~locals e
            in
            ())
          emitters
      in
      Err.return { canonical_shape; canonical_partition; locals; emitters }

let finish ~max_size ~max_depth ~canonical_shape ~emitters =
  Region_program.Builder.of_fn (fun state locals ->
      ( create ~max_size ~max_depth ~canonical_shape ~locals:(List.rev locals)
          ~emitters,
        state ))

let pp fmt t =
  let names =
    List.mapi
      (fun i (local : Region_local.t) ->
        (local.Region_local.id, Fmt.str "l%d" i))
      t.locals
    |> List.to_seq |> Expr.Local_var.Map.of_seq
  in
  let local_name id = Expr.Local_var.Map.find_opt id names in
  Fmt.pf fmt "canonical [%a]" Region_partition.pp t.canonical_partition;
  List.iter
    (fun (local : Region_local.t) ->
      Fmt.pf fmt "@\n  let %s : %a = "
        (Option.value ~default:"?" (local_name local.Region_local.id))
        Region_local.Shape.pp
        (Region_local.Shape.of_rhs local.Region_local.rhs);
      match local.Region_local.rhs with
      | Region_local.Rhs.Scan s -> Expr.Pp.scan_open ~names:local_name fmt s
      | Region_local.Rhs.Scalar _ | Region_local.Rhs.Vector _ ->
          Expr.Pp.value_open ~names:local_name fmt
            (Region_local.Rhs.value local.Region_local.rhs))
    t.locals;
  List.iteri
    (fun i (e : Emitter.t) ->
      Fmt.pf fmt "@\n  emitter %d [%a] = %a" i Region_partition.pp
        e.Emitter.partition
        (Expr.Pp.value_open ~names:local_name)
        e.Emitter.output)
    t.emitters

type region_group = t

module Ref = struct
  type t = Grouped of region_group * int | Solo of Region_program.t

  (* Safe by construction: every [Grouped] value in this codebase is built
     with an ordinal drawn from the SAME group's own emitter list (see the
     module doc on grouping identity above), never a foreign or
     out-of-range one. *)
  let sources = function
    | Solo p -> Region_program.Fold.sources p
    | Grouped (g, i) -> Option.get (sources g i)

  let max_depth = function
    | Solo p -> Region_program.Fold.max_depth p
    | Grouped (g, i) -> Option.get (max_depth g i)

  let intrinsic_sources = function
    | Solo p -> Region_program.Fold.intrinsic_sources p
    | Grouped (g, i) -> Option.get (intrinsic_sources g i)

  let binders = function
    | Solo p -> Region_program.Fold.binders p
    | Grouped (g, i) -> Option.get (binders g i)

  let intrinsics = function
    | Solo p -> Region_program.Fold.intrinsics p
    | Grouped (g, i) -> Option.get (intrinsics g i)

  (* A [Solo] program is re-[check]ed here too, not merely returned: a
     [Region_program.t] can reach a [Kernel.Value.t]/[Stage_program.Stage.t]
     via [with_output] (a raw record update with no check of its own) or a
     hand-built [Kernel.t]/[Stage_program.t] (both public, untrusted
     records), so skipping this would silently drop the re-validation every
     caller of [project] historically got from calling
     [Region_program.check] unconditionally. *)
  let project ~max_size ~max_depth = function
    | Solo p ->
        let open Err.Syntax in
        let+ () =
          Err.map_error
            (fun e -> `Program e)
            (Region_program.check ~max_size ~max_depth p)
        in
        p
    | Grouped (g, i) -> project ~max_size ~max_depth g i

  let check ~max_size ~max_depth t =
    let open Err.Syntax in
    let+ (_ : Region_program.t) = project ~max_size ~max_depth t in
    ()

  let pixel_expression = function
    | Solo p -> Region_program.pixel_expression p
    | Grouped _ -> None

  let pp fmt = function
    | Solo p -> Region_program.pp fmt p
    | Grouped (g, i) -> Fmt.pf fmt "group emitter %d of@ %a" i pp g
end

(* A maximal run of consecutive items sharing one physically-identical
   [Region_group.t] (compared with [==], never structural equality -- design
   record §3.1's "do not use structural equality... as a cache identity"), or
   a single ordinary item. Factored out of [Stage_program.ground]'s own
   [runs_of_stages] (project step 19's first user of this shape) now that
   [Kernel_eval.execute] needs the identical partition over [Kernel.Value.t]
   -- generic over the item type via [~computation], so neither caller
   restates the run-detection fold. *)
module Run = struct
  type 'a t = Solo of 'a | Group of region_group * (int * 'a) list

  let duplicate_ordinal = function
    | Solo _ -> None
    | Group (_, members) ->
        let rec find seen = function
          | [] -> None
          | (ordinal, _) :: rest ->
              if List.mem ordinal seen then Some ordinal
              else find (ordinal :: seen) rest
        in
        find [] members
end

let runs ~computation items =
  let close = function
    | None -> Fun.id
    | Some (Run.Solo _ as r) -> List.cons r
    | Some (Run.Group (g, members)) ->
        List.cons (Run.Group (g, List.rev members))
  in
  let rec go acc current = function
    | [] -> List.rev (close current acc)
    | item :: rest -> (
        match (computation item, current) with
        | Ref.Solo _, _ -> go (close current acc) (Some (Run.Solo item)) rest
        | Ref.Grouped (g, ord), Some (Run.Group (g', members)) when g == g' ->
            go acc (Some (Run.Group (g', (ord, item) :: members))) rest
        | Ref.Grouped (g, ord), _ ->
            go (close current acc) (Some (Run.Group (g, [ (ord, item) ]))) rest)
  in
  go [] None items
