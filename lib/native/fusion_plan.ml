(* See fusion_plan.mli. *)

module Rejection = struct
  type t =
    | Multiple_uses of { producer : Tensor_id.t; at_least : int }
    | Intrinsic_use of Kernel.Use.t
    | Reducing_consumer of Kernel.Use.t
    | Non_pointwise_use of {
        use : Kernel.Use.t;
        reason : Kernel_elab.error Err.Error.t;
      }
    | Overlaps_selected of { rejected : Kernel.Use.t; selected : Kernel.Use.t }
    | Budget_exceeded of {
        use : Kernel.Use.t;
        error : Kernel_elab.error Err.Error.t;
      }

  let pp fmt = function
    | Multiple_uses { producer; at_least } ->
        Fmt.pf fmt "%a has at least %d ordinary uses" Tensor_id.pp producer
          at_least
    | Intrinsic_use u ->
        Fmt.pf fmt "%a is reached through an intrinsic descriptor" Kernel.Use.pp
          u
    | Reducing_consumer u ->
        Fmt.pf fmt "%a has a reducing or intrinsic-bearing consumer"
          Kernel.Use.pp u
    | Non_pointwise_use { use; reason } ->
        Fmt.pf fmt "%a is not a pointwise site (%a)" Kernel.Use.pp use
          (Core.Pretty.error_kind Kernel_elab.pp_error)
          reason
    | Overlaps_selected { rejected; selected } ->
        Fmt.pf fmt "%a overlaps the already-selected %a" Kernel.Use.pp rejected
          Kernel.Use.pp selected
    | Budget_exceeded { use; error } ->
        Fmt.pf fmt "%a exceeds the budget (%a)" Kernel.Use.pp use
          (Core.Pretty.error_kind Kernel_elab.pp_error)
          error
end

module Decision = struct
  type t =
    | Virtualize of { use : Kernel.Use.t; also_stored : bool }
    | Reject of { producer : Tensor_id.t; reason : Rejection.t }

  let pp fmt = function
    | Virtualize { use; also_stored } ->
        Fmt.pf fmt "virtualize %a%s" Kernel.Use.pp use
          (if also_stored then " (also stored)" else "")
    | Reject { producer; reason } ->
        Fmt.pf fmt "reject %a: %a" Tensor_id.pp producer Rejection.pp reason
end

type t = {
  kernel : Kernel.t;
  virtual_uses : Kernel.Use.Set.t;
  stores : Tensor_id.Set.t;
}

let all_values (k : Kernel.t) =
  List.fold_left
    (fun s (v : Kernel.Value.t) -> Tensor_id.Set.add v.Kernel.Value.id s)
    Tensor_id.Set.empty k.Kernel.values

let default k =
  { kernel = k; virtual_uses = Kernel.Use.Set.empty; stores = all_values k }

(* [stores] is derived, never accumulated: a value is stored unless it is the
   producer of a virtual use. That is what gives placement closure — any load
   surviving elaboration reads a buffer. The proof needs the unique-use rule,
   not the one-edge representation: inlining P into C leaves P's own loads of
   some Q, and were Q virtualized on another edge it would have two ordinary
   uses (Q->P and Q->D) and be rejected as [Multiple_uses]. *)
let stores_for k virtual_uses =
  Kernel.Use.Set.fold
    (fun u s -> Tensor_id.Set.remove u.Kernel.Use.producer s)
    virtual_uses (all_values k)

(* Id -> topological position, built once. A linear scan per lookup would put
   another multiplicative factor on the report's sort, for the same reason the
   per-candidate rescans did on planning. *)
let ordinals (k : Kernel.t) =
  let t = Hashtbl.create 64 in
  List.iteri
    (fun i (v : Kernel.Value.t) ->
      Hashtbl.replace t (Tensor_id.to_int v.Kernel.Value.id) i)
    k.Kernel.values;
  fun id ->
    Option.value (Hashtbl.find_opt t (Tensor_id.to_int id)) ~default:max_int

(* Disjoint one-edge groups: a value is a producer in at most one virtual use
   and a consumer in at most one, and the two roles do not meet. That is what
   makes a single [substitute_loads] pass correct — inserted subtrees are not
   re-traversed, so a chain would otherwise leave a load of another virtual
   value behind and the budget would undercount the full expansion.

   Answered by endpoint OCCUPANCY, not by a filter over the selected set:
   disjointness is a question about the two ids, so a table from an occupied id
   to the use that claimed it answers it directly, where filtering every
   selected edge per candidate made selection quadratic in the selected
   count. *)
let overlapping occupied (u : Kernel.Use.t) =
  match Hashtbl.find_opt occupied (Tensor_id.to_int u.Kernel.Use.producer) with
  | Some s -> Some s
  | None -> Hashtbl.find_opt occupied (Tensor_id.to_int u.Kernel.Use.consumer)

let occupy occupied (u : Kernel.Use.t) =
  Hashtbl.replace occupied (Tensor_id.to_int u.Kernel.Use.producer) u;
  Hashtbl.replace occupied (Tensor_id.to_int u.Kernel.Use.consumer) u

let plan (k : Kernel.t) =
  let load_uses = Kernel.load_uses k in
  let all_uses = Kernel.uses k in
  (* Everything a candidate decision needs is collected before any candidate is
     considered. Recomputing any of it per candidate made the planner's work
     multiplicative in AST size and edge count, so the accepted IR limits gave
     no practical time bound: a kernel may hold thousands of values with bodies
     near the body budget, and each edge would re-walk all of them.

     The load-derived half — ordered candidates, per-(consumer, producer)
     occurrence, the saturated load count — lives in [Kernel_elab.Analysis],
     which OWNS it precisely so [site_in] cannot be handed fabricated evidence.
     The pass below covers the remaining queries.

     What is still per attempted elaboration, deliberately: the freshening,
     substitution and post-rewrite budget check in [elaborate_site]. That is the
     work fusion is actually for. *)
  let analysis = Kernel_elab.Analysis.of_kernel k in
  let intrinsic_edges = Hashtbl.create 16 in
  let pointwise = Hashtbl.create 64 in
  List.iter
    (fun (v : Kernel.Value.t) ->
      let cid = Tensor_id.to_int v.Kernel.Value.id in
      List.iter
        (fun s ->
          let producer = Expr_bridge.id_of_source s in
          let pid = Tensor_id.to_int producer in
          if not (Hashtbl.mem intrinsic_edges pid) then
            Hashtbl.add intrinsic_edges pid
              { Kernel.Use.producer; consumer = v.Kernel.Value.id })
        (Expr.Fold.intrinsic_sources v.Kernel.Value.body);
      Hashtbl.replace pointwise cid
        (Expr.Fold.binders v.Kernel.Value.body = []
        && Expr.Fold.intrinsics v.Kernel.Value.body = 0))
    k.Kernel.values;
  let load_count id = Kernel_elab.Analysis.load_count analysis id in
  let intrinsic_edge id =
    Hashtbl.find_opt intrinsic_edges (Tensor_id.to_int id)
  in
  let consumer_is_pointwise (v : Kernel.Value.t) =
    Option.value
      (Hashtbl.find_opt pointwise (Tensor_id.to_int v.Kernel.Value.id))
      ~default:false
  in
  let outputs =
    List.fold_left
      (fun s (o : Kernel.Output.t) -> Tensor_id.Set.add o.Kernel.Output.value s)
      Tensor_id.Set.empty k.Kernel.outputs
  in
  (* Consumers in topological order; within one consumer, incoming candidates in
     first-occurrence order in [Fold.loads] — lexical in the body, so the plan
     depends on the expression rather than on a set's iteration order. *)
  (* Load-order candidates first; then this consumer's remaining dependency
     edges, which are the intrinsic-only ones.

     Enumerating only [load_uses] would leave [Intrinsic_use] unreachable — a
     check that can never fail — and would silently produce NO decision for a
     producer whose only consumer reads it through a descriptor, contradicting
     "one decision per produced value with a downstream dependency". Those edges
     can only ever be rejected, so their order among themselves affects the
     report alone; producer id order makes it deterministic. *)
  (* The intrinsic-only remainder, grouped by consumer in ONE pass over the edge
     set. Converting [all_uses] to a list and filtering it per value scanned
     every edge once per value — including the ordinary-load edges already in
     [candidate_sites] — and then asked [List.mem] against them, so candidate
     construction stayed O(values x edges) no matter how much body metadata had
     been precomputed. A wide kernel can hold very many distinct edges under the
     per-value limits, which is exactly where that bites. *)
  let extra = Hashtbl.create 16 in
  Kernel.Use.Set.iter
    (fun (u : Kernel.Use.t) ->
      if not (Kernel.Use.Set.mem u load_uses) then
        let cid = Tensor_id.to_int u.Kernel.Use.consumer in
        Hashtbl.replace extra cid
          (u :: Option.value (Hashtbl.find_opt extra cid) ~default:[]))
    all_uses;
  let candidates_of (v : Kernel.Value.t) =
    let cid = Tensor_id.to_int v.Kernel.Value.id in
    let by_load = Kernel_elab.Analysis.candidates analysis v.Kernel.Value.id in
    (* [Use.Set.iter] is ascending, so the accumulated list is descending;
       reversing restores producer-id order, which is what makes the report
       deterministic for edges that can only ever be rejected. *)
    let rest =
      List.rev (Option.value (Hashtbl.find_opt extra cid) ~default:[])
    in
    by_load @ rest
  in
  let occupied = Hashtbl.create 16 in
  let selected, rejections =
    List.fold_left
      (fun acc (v : Kernel.Value.t) ->
        List.fold_left
          (fun (selected, rejections) (u : Kernel.Use.t) ->
            let reject r =
              (selected, (u.Kernel.Use.producer, r) :: rejections)
            in
            (* Precedence, first failing check wins: structural facts about the
               producer, then per-edge coordinate facts, then selection
               outcomes, then the budget — the only expensive one, since it
               elaborates. *)
            match intrinsic_edge u.Kernel.Use.producer with
            | Some e -> reject (Rejection.Intrinsic_use e)
            | None -> (
                let n = load_count u.Kernel.Use.producer in
                if n > 1 then
                  reject
                    (Rejection.Multiple_uses
                       { producer = u.Kernel.Use.producer; at_least = n })
                else if not (consumer_is_pointwise v) then
                  reject (Rejection.Reducing_consumer u)
                else
                  (* The consumer's loads come from the pass above, so a
                     candidate costs no body fold, and the resulting [Site.t]
                     carries the resolved endpoints into elaboration — a legal
                     edge used to re-fold its consumer and repeat both lookups. *)
                  match Kernel_elab.site_in analysis u with
                  | Error e ->
                      reject
                        (Rejection.Non_pointwise_use { use = u; reason = e })
                  | Ok prepared -> (
                      match overlapping occupied u with
                      | Some s ->
                          reject
                            (Rejection.Overlaps_selected
                               { rejected = u; selected = s })
                      | None -> (
                          match Kernel_elab.elaborate_site prepared with
                          | Error e ->
                              reject
                                (Rejection.Budget_exceeded
                                   { use = u; error = e })
                          | Ok _ ->
                              occupy occupied u;
                              (Kernel.Use.Set.add u selected, rejections)))))
          acc (candidates_of v))
      (Kernel.Use.Set.empty, []) k.Kernel.values
  in
  let stores = stores_for k selected in
  (* One decision per produced value with a downstream dependency, in producer
     topological order. *)
  let decisions =
    let virt =
      Kernel.Use.Set.fold
        (fun u acc ->
          ( u.Kernel.Use.producer,
            Decision.Virtualize
              {
                use = u;
                also_stored = Tensor_id.Set.mem u.Kernel.Use.producer outputs;
              } )
          :: acc)
        selected []
    in
    let selected_producers = Hashtbl.create 16 in
    Kernel.Use.Set.iter
      (fun (u : Kernel.Use.t) ->
        Hashtbl.replace selected_producers
          (Tensor_id.to_int u.Kernel.Use.producer)
          ())
      selected;
    let rej =
      List.filter_map
        (fun (producer, reason) ->
          (* A producer-id set built once. [Set.exists] per rejection made the
             report O(rejections x selected), which bites even when every
             candidate was rejected before overlap checking ever ran. *)
          if Hashtbl.mem selected_producers (Tensor_id.to_int producer) then
            None
          else Some (producer, Decision.Reject { producer; reason }))
        (List.rev rejections)
    in
    (* Deduplicate to one decision per producer, keeping the first reason. *)
    let seen = Hashtbl.create 16 in
    List.filter
      (fun (id, _) ->
        let k = Tensor_id.to_int id in
        if Hashtbl.mem seen k then false
        else (
          Hashtbl.add seen k ();
          true))
      (virt @ rej)
    |> (let ordinal = ordinals k in
        List.sort (fun (a, _) (b, _) -> compare (ordinal a) (ordinal b)))
    |> List.map snd
  in
  ( {
      kernel = k;
      virtual_uses = selected;
      stores =
        (* An externally live producer stays stored AS WELL as being virtual for
           its consumer. Both facts, which is why placement is two sets. *)
        Tensor_id.Set.union stores (Tensor_id.Set.inter outputs (all_values k));
    },
    decisions )
