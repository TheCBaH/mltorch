(* See kernel_elab.mli. *)

type error =
  [ `Body of Kernel.Body_error.t
  | `Not_a_dependency of Kernel.Use.t
  | `Regional_computation of Kernel.Use.t
  | `Unknown_use of Kernel.Use.t
  | `Unsupported_use of Kernel.Use.t ]

let pp_error fmt : [< error ] -> unit = function
  | `Body { Kernel.Body_error.at; error } ->
      Fmt.pf fmt "%a: %a" Tensor_id.pp at Region_program.pp_error error
  | `Not_a_dependency u ->
      Fmt.pf fmt "%a is not an ordinary-load edge" Kernel.Use.pp u
  | `Regional_computation u ->
      Fmt.pf fmt "%a has a regional computation" Kernel.Use.pp u
  | `Unknown_use u ->
      (* Includes a boundary input as producer: the kernel defines that id, but
         not as a VALUE, so there is no body or result conversion to
         substitute. *)
      Fmt.pf fmt "%a does not name a value at both ends" Kernel.Use.pp u
  | `Unsupported_use u ->
      Fmt.pf fmt "%a is outside the supported site class" Kernel.Use.pp u

let extent (sg : Tensor_sig.t) axis =
  Dim.to_int (Vec6.get sg.Tensor_sig.shape axis)

(* Per axis: the plain output variable with equal extents, or [zero] where the
   producer extent is 1. A [zero] on a larger extent is a constant slice, not
   broadcasting — executable, but not the class this admits. *)
let pointwise ~(producer : Tensor_sig.t) ~(consumer : Tensor_sig.t) coord =
  List.for_all
    (fun axis ->
      match Expr.Coord.get coord axis with
      | Expr.Index.Output a ->
          Expr.Axis.equal a axis && extent producer axis = extent consumer axis
      | Expr.Index.Zero -> extent producer axis = 1
      | _ -> false)
    Expr.Axis.all

module Site = struct
  (* A VALIDATED edge: both endpoints resolved and the single admissible load
     coordinate found. Opaque, so the only way to hold one is to have passed
     [site], and [elaborate_site] can then rewrite without re-resolving or
     re-folding anything. *)
  type t = {
    use : Kernel.Use.t;
    producer : Kernel.Value.t;
    consumer : Kernel.Value.t;
    coord : Expr.Role.Position.t Expr.Index.t Expr.Coord.t;
    limits : Kernel.Limits.t;
  }

  let use s = s.use
  let coord s = s.coord
end

(* Occurrences of one producer inside one consumer. Only the distinction the
   rule turns on is kept — none, exactly one at a coordinate, or several — so
   admission reads a summary instead of filtering the consumer's whole load
   list. Filtering per candidate made a consumer with M distinct producers do
   O(M^2) list work, and site legality deliberately precedes the overlap check,
   so every later candidate paid it in full before being rejected. *)
type occurrence =
  | One of Expr.Role.Position.t Expr.Index.t Expr.Coord.t
  | Several

(* The one legality rule, over already-extracted data. Both entry points below
   call it, so the predicate cannot differ between the planner and a direct
   caller — and neither can pay for the extraction twice. *)
let admit ~limits ~(u : Kernel.Use.t) ~(producer : Kernel.Value.t)
    ~(consumer : Kernel.Value.t) ~occurrence =
  if
    Option.is_none (Kernel.pixel_expression producer)
    || Option.is_none (Kernel.pixel_expression consumer)
  then Err.fail (`Regional_computation u)
  else
    match occurrence with
    | None -> Err.fail (`Not_a_dependency u)
    | Some Several -> Err.fail (`Unsupported_use u)
    | Some (One coord) ->
        if
          pointwise ~producer:producer.Kernel.Value.sg
            ~consumer:consumer.Kernel.Value.sg coord
        then Err.return { Site.use = u; producer; consumer; coord; limits }
        else Err.fail (`Unsupported_use u)

module Analysis = struct
  (* The extracted evidence, OWNED. Derived from a [Kernel.t] and reachable only
     through [of_kernel], so a caller cannot fabricate it or mix data belonging
     to another consumer.

     The previous shape took the extracted load list as an argument and treated
     it as proof, which proved nothing: one fabricated pointwise entry yielded a
     [Site.t] for a pair that is not a dependency at all, and [elaborate_site]
     then rewrote the consumer's REAL body — substituting nothing and reporting
     the unchanged body as a successful elaboration, or substituting every real
     occurrence of a multi-site edge the checks were there to refuse. A private
     type whose constructor accepts unchecked data is not private. *)
  type t = {
    kernel : Kernel.t;
    occurrences : (int * int, occurrence) Hashtbl.t;
        (** (consumer, producer) -> how the producer appears in that consumer *)
    order : (int, Kernel.Use.t list) Hashtbl.t;
        (** consumer -> its load edges, in first-occurrence order *)
    counts : (int, int) Hashtbl.t;
        (** producer -> ordinary loads, capped at 2 *)
  }

  let of_kernel (k : Kernel.t) =
    let occurrences = Hashtbl.create 64 in
    let order = Hashtbl.create 64 in
    let counts = Hashtbl.create 64 in
    List.iter
      (fun (v : Kernel.Value.t) ->
        let cid = Tensor_id.to_int v.Kernel.Value.id in
        let rev = ref [] in
        List.iter
          (fun (s, c) ->
            let producer = Expr_bridge.id_of_source s in
            let pid = Tensor_id.to_int producer in
            (* Saturating: the cross-body total is an [int], and the per-value
               limits permit an aggregate past 2^31, which wraps negative under
               js_of_ocaml. Legality only asks one versus more than one. *)
            let n = Option.value (Hashtbl.find_opt counts pid) ~default:0 in
            if n < 2 then Hashtbl.replace counts pid (n + 1);
            match Hashtbl.find_opt occurrences (cid, pid) with
            | Some Several -> ()
            | Some (One _) -> Hashtbl.replace occurrences (cid, pid) Several
            | None ->
                Hashtbl.replace occurrences (cid, pid) (One c);
                if Option.is_some (Kernel.value k producer) then
                  rev :=
                    { Kernel.Use.producer; consumer = v.Kernel.Value.id }
                    :: !rev)
          (Region_program.Fold.loads v.Kernel.Value.computation);
        Hashtbl.replace order cid (List.rev !rev))
      k.Kernel.values;
    { kernel = k; occurrences; order; counts }

  let kernel a = a.kernel

  let candidates a consumer =
    Option.value
      (Hashtbl.find_opt a.order (Tensor_id.to_int consumer))
      ~default:[]

  let load_count a producer =
    Option.value
      (Hashtbl.find_opt a.counts (Tensor_id.to_int producer))
      ~default:0
end

let site_in (a : Analysis.t) (u : Kernel.Use.t) =
  let k = Analysis.kernel a in
  match
    (Kernel.value k u.Kernel.Use.producer, Kernel.value k u.Kernel.Use.consumer)
  with
  | None, _ | _, None -> Err.fail (`Unknown_use u)
  | Some producer, Some consumer ->
      admit ~limits:k.Kernel.limits ~u ~producer ~consumer
        ~occurrence:
          (Hashtbl.find_opt a.Analysis.occurrences
             ( Tensor_id.to_int u.Kernel.Use.consumer,
               Tensor_id.to_int u.Kernel.Use.producer ))

(* One producer's occurrences inside ONE body, summarised to the distinction the
   rule turns on. Shared by neither path's indexing — [Analysis] builds its table
   over every consumer at once, this walks a single body — but feeding the same
   [admit], so the two extractions can be compared and the rule still cannot
   differ between them. *)
let summarize ~src loads =
  List.fold_left
    (fun acc (s, c) ->
      if Expr.Source.equal s src then
        match acc with None -> Some (One c) | Some _ -> Some Several
      else acc)
    None loads

(* Self-contained and SITE-LOCAL: it resolves the two endpoints and folds only
   the named consumer. Routing this through [Analysis.of_kernel] instead would
   make one-edge elaboration proportional to the whole kernel AST — every body
   folded and three tables built — which is the planner's cost, incurred because
   it checks many edges, and not something a standalone one-edge call should
   pay. A caller checking many edges builds one [Analysis.t] and uses
   [site_in]. *)
let site (k : Kernel.t) (u : Kernel.Use.t) =
  match
    (Kernel.value k u.Kernel.Use.producer, Kernel.value k u.Kernel.Use.consumer)
  with
  | None, _ | _, None -> Err.fail (`Unknown_use u)
  | Some producer, Some consumer ->
      admit ~limits:k.Kernel.limits ~u ~producer ~consumer
        ~occurrence:
          (summarize
             ~src:(Expr_bridge.source_of_id u.Kernel.Use.producer)
             (Region_program.Fold.loads consumer.Kernel.Value.computation))

let elaborate_site (s : Site.t) =
  let open Err.Syntax in
  let u = s.Site.use in
  let producer = s.Site.producer and consumer = s.Site.consumer in
  let src = Expr_bridge.source_of_id u.Kernel.Use.producer in
  (* ONE builder namespace for the whole result. The destination is freshened
     first and every inserted fragment is minted from the state this threads,
     which is what keeps the producer's binders off the consumer's. Freshening
     the producer from [Builder.initial] instead would have it mint ordinal 0
     again and reintroduce, at the moment of composing, the collision [freshen]
     exists to prevent. *)
  let composed =
    let open Expr.Builder.Syntax in
    let* dest =
      Expr.Rewrite.freshen (Option.get (Kernel.pixel_expression consumer))
    in
    Expr.Rewrite.substitute_loads
      (fun s load_coord ->
        if Expr.Source.equal s src then
          Some
            (let+ body =
               Expr.Rewrite.freshen
                 (Option.get (Kernel.pixel_expression producer))
             in
             (* The producer's result conversion, applied exactly once —
                [eval_value]'s syntactic counterpart. Dropping it is a semantic
                change, not an optimisation, and is invisible while the producer
                is stored because the f32 store rounds anyway. It becomes
                load-bearing precisely here. *)
             Kernel.Result_conversion.apply producer.Kernel.Value.result
               (Expr.Rewrite.substitute_output load_coord body))
        else None)
      dest
  in
  let body = Expr.Builder.run composed in
  let+ () =
    Err.map_error
      (fun e ->
        `Body { Kernel.Body_error.at = u.Kernel.Use.consumer; error = `Expr e })
      (Expr.Check.value ~max_size:s.Site.limits.Kernel.Limits.max_size
         ~max_depth:s.Site.limits.Kernel.Limits.max_depth body)
  in
  body

(* The self-contained public form: validate, then rewrite. A caller that already
   holds a [Site.t] uses [elaborate_site] and pays for neither again. *)
let elaborate (k : Kernel.t) (u : Kernel.Use.t) =
  let open Err.Syntax in
  let* s = site k u in
  elaborate_site s
