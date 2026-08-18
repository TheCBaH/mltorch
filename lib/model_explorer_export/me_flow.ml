(* The flow spine. See the .mli. *)

module Pass_execution = struct
  type t = { layer : Me_ids.Layer.t; exec : Pass.Exec_id.t }

  (* Structural, over the whole pair. The layer comes first because that is the
     half that makes it unique: two dialects' ordinals are independently dense,
     so comparing executions alone would call two different transitions
     equal. *)
  let compare a b =
    match
      String.compare
        (Me_ids.Layer.to_string a.layer)
        (Me_ids.Layer.to_string b.layer)
    with
    | 0 -> Stdlib.compare a.exec b.exec
    | c -> c

  let pp fmt t =
    Fmt.pf fmt "%s:%a" (Me_ids.Layer.to_string t.layer) Pass.Exec_id.pp t.exec

  module Map = Map.Make (struct
    type nonrec t = t

    let compare = compare
  end)
end

module State = struct
  type t = {
    id : string;
    graph : string;
    view : string;
    layer : Me_ids.Layer.t;
    label : string;
    produced_by : string option;
  }
end

module Transition = struct
  module Kind_tag = struct
    type t = Import | Pass | Pack | Cross_dialect | Adapt

    let to_string = function
      | Import -> "import"
      | Pass -> "pass"
      | Pack -> "pack"
      | Cross_dialect -> "cross_dialect"
      | Adapt -> "adapt"

    let next = function
      | Import -> Some Pass
      | Pass -> Some Pack
      | Pack -> Some Cross_dialect
      | Cross_dialect -> Some Adapt
      | Adapt -> None

    let all =
      let rec walk acc c =
        match next c with
        | None -> List.rev (c :: acc)
        | Some n -> walk (c :: acc) n
      in
      walk [] Import
  end

  type kind = Import | Pass of Pass_execution.t | Pack | Cross_dialect | Adapt

  let tag : kind -> Kind_tag.t = function
    | Import -> Kind_tag.Import
    | Pass _ -> Kind_tag.Pass
    | Pack -> Kind_tag.Pack
    | Cross_dialect -> Kind_tag.Cross_dialect
    | Adapt -> Kind_tag.Adapt

  let kind_name k = Kind_tag.to_string (tag k)

  type t = {
    id : string;
    before : string;
    after : string;
    kind : kind;
    comparison : string option;
  }
end

type t = {
  states : State.t list;
  transitions : Transition.t list;
  graph : string;
}

(* Own modules, per the record-namespace convention: both payloads have a
   [transition] field, and distinct namespaces are how this repo keeps labels
   unique rather than silencing warning 30. *)
module Illegal_transition = struct
  type t = {
    transition : string;
    before : Me_ids.Layer.t;
    kind : Transition.Kind_tag.t;
    after : Me_ids.Layer.t;
  }
end

module Pass_layer_disagreement = struct
  type t = {
    transition : string;
    execution : Me_ids.Layer.t;
    before : Me_ids.Layer.t;
    after : Me_ids.Layer.t;
  }
end

type error =
  [ `Duplicate_state of string
  | `Duplicate_transition of string
  | `Unknown_state of string
  | `No_root
  | `Multiple_roots of int
  | `Unreachable_state of string
  | `Cycle of string
  | `Multiple_producers of string
  | `Producer_disagrees of string
  | `Illegal_transition of Illegal_transition.t
  | `Pass_layer_disagrees of Pass_layer_disagreement.t
  | `Duplicate_pass_execution of Pass_execution.t
  | Me_limits.over_limit_error ]

let pp_error fmt : [< error ] -> unit = function
  | `Duplicate_state id -> Fmt.pf fmt "duplicate flow state %s" id
  | `Duplicate_transition id -> Fmt.pf fmt "duplicate flow transition %s" id
  | `Unknown_state id -> Fmt.pf fmt "transition names unknown state %s" id
  | `No_root -> Fmt.pf fmt "no pt2 root state"
  | `Multiple_roots n -> Fmt.pf fmt "%d root states, expected one" n
  | `Unreachable_state id ->
      Fmt.pf fmt "state %s is unreachable from the root" id
  | `Cycle id -> Fmt.pf fmt "flow is cyclic at state %s" id
  | `Multiple_producers id ->
      Fmt.pf fmt "state %s has more than one producer" id
  | `Producer_disagrees id ->
      Fmt.pf fmt "state %s names a producer that does not produce it" id
  | `Illegal_transition { Illegal_transition.transition; before; kind; after }
    ->
      Fmt.pf fmt "transition %s crosses layers illegally: %s -%s-> %s"
        transition
        (Me_ids.Layer.to_string before)
        (Transition.Kind_tag.to_string kind)
        (Me_ids.Layer.to_string after)
  | `Pass_layer_disagrees
      { Pass_layer_disagreement.transition; execution; before; after } ->
      Fmt.pf fmt
        "transition %s carries an execution from another layer: %s, between %s \
         and %s"
        transition
        (Me_ids.Layer.to_string execution)
        (Me_ids.Layer.to_string before)
        (Me_ids.Layer.to_string after)
  | `Duplicate_pass_execution e ->
      Fmt.pf fmt "pass execution %a occurs more than once" Pass_execution.pp e
  | `Over_limit o -> Me_limits.Over_limit.pp fmt o

let count = Me_limits.check ~scope:Me_limits.Scope.Flow

(* Which conversions exist, stated once. [Pass] and [Pack] stay inside a layer;
   the three that leave one are named individually rather than by a rule,
   because the dataflow BRANCHES at canonical Native -- to Native4D by
   conversion and to the symbolic stages by adaptation -- and a rule of the form
   "the next layer" cannot express a branch. *)
let legal_triples =
  let open Me_ids.Layer in
  let open Transition.Kind_tag in
  [
    (Pt2, Import, Native);
    (Native, Pass, Native);
    (Native, Pack, Native);
    (Native, Cross_dialect, Native4d);
    (Native4d, Pass, Native4d);
    (Native4d, Pack, Native4d);
    (Native, Adapt, Symbolic);
    (Symbolic, Pass, Symbolic);
    (Symbolic, Adapt, Kernel);
    (Kernel, Pass, Kernel);
  ]

let is_legal ~before ~kind ~after =
  List.exists
    (fun (b, k, a) -> b = before && k = kind && a = after)
    legal_triples

let validate ~limits flow =
  let open Err.Syntax in
  (* --- the two aggregates, FIRST ---

     Before the walks below, which are linear in these counts: a bound checked
     after the work it bounds is not a bound. *)
  let* () =
    let* () =
      count Me_limits.Field.States (List.length flow.states)
        ~ceiling:limits.Me_limits.Limits.max_states
    in
    count Me_limits.Field.Transitions
      (List.length flow.transitions)
      ~ceiling:limits.Me_limits.Limits.max_transitions
  in
  (* --- unique ids, and a lookup --- *)
  let* states =
    Err.List.fold_left
      (fun acc (s : State.t) ->
        if Hashtbl.mem acc s.State.id then
          Err.fail (`Duplicate_state s.State.id)
        else begin
          Hashtbl.add acc s.State.id s;
          Err.return acc
        end)
      (Hashtbl.create 16) flow.states
  in
  (* Uniqueness only -- nothing below looks a transition up by id, and a table
     built to be discarded would read as one that is. *)
  let* (_ : (string, Transition.t) Hashtbl.t) =
    Err.List.fold_left
      (fun acc (t : Transition.t) ->
        if Hashtbl.mem acc t.Transition.id then
          Err.fail (`Duplicate_transition t.Transition.id)
        else begin
          Hashtbl.add acc t.Transition.id t;
          Err.return acc
        end)
      (Hashtbl.create 16) flow.transitions
  in
  let state id =
    match Hashtbl.find_opt states id with
    | Some s -> Err.return s
    | None -> Err.fail (`Unknown_state id)
  in
  (* --- endpoints resolve, layers agree, executions are unique --- *)
  let seen_exec = Hashtbl.create 16 in
  let* () =
    Err.List.iter
      (fun (t : Transition.t) ->
        let* before = state t.Transition.before in
        let* after = state t.Transition.after in
        let kind = Transition.tag t.Transition.kind in
        let* () =
          if is_legal ~before:before.State.layer ~kind ~after:after.State.layer
          then Err.return ()
          else
            Err.fail
              (`Illegal_transition
                 {
                   Illegal_transition.transition = t.Transition.id;
                   before = before.State.layer;
                   kind;
                   after = after.State.layer;
                 })
        in
        match t.Transition.kind with
        | Transition.Pass e ->
            (* The execution's own layer must equal BOTH endpoints', not just
               one: a pass stays inside its layer, so the two are the same
               check only once the triple above has passed. *)
            let* () =
              if
                e.Pass_execution.layer = before.State.layer
                && e.Pass_execution.layer = after.State.layer
              then Err.return ()
              else
                Err.fail
                  (`Pass_layer_disagrees
                     {
                       Pass_layer_disagreement.transition = t.Transition.id;
                       execution = e.Pass_execution.layer;
                       before = before.State.layer;
                       after = after.State.layer;
                     })
            in
            (* Keyed by the RENDERING, which [Pass_execution.pp] makes injective
               over the pair; the error carries the value itself. *)
            let key = Core.Pretty.to_string Pass_execution.pp e in
            if Hashtbl.mem seen_exec key then
              Err.fail (`Duplicate_pass_execution e)
            else begin
              Hashtbl.add seen_exec key ();
              Err.return ()
            end
        | _ -> Err.return ())
      flow.transitions
  in
  (* --- at most one producer per state, agreeing with [produced_by] --- *)
  let producer = Hashtbl.create 16 in
  let* () =
    Err.List.iter
      (fun (t : Transition.t) ->
        if Hashtbl.mem producer t.Transition.after then
          Err.fail (`Multiple_producers t.Transition.after)
        else begin
          Hashtbl.add producer t.Transition.after t.Transition.id;
          Err.return ()
        end)
      flow.transitions
  in
  let* () =
    Err.List.iter
      (fun (s : State.t) ->
        match (s.State.produced_by, Hashtbl.find_opt producer s.State.id) with
        | None, None -> Err.return ()
        | Some a, Some b when String.equal a b -> Err.return ()
        | _ -> Err.fail (`Producer_disagrees s.State.id))
      flow.states
  in
  (* --- exactly one Pt2 root --- *)
  let roots =
    List.filter
      (fun (s : State.t) ->
        s.State.produced_by = None && s.State.layer = Me_ids.Layer.Pt2)
      flow.states
  in
  let* root =
    match roots with
    | [ r ] -> Err.return r
    | [] -> Err.fail `No_root
    | l -> Err.fail (`Multiple_roots (List.length l))
  in
  (* --- reachability, with a termination guard ---

     A depth-first walk colouring grey on entry and black on exit. Every state
     must come out black, which is reachability.

     [`Cycle] is NOT reachable through the checks above it, and saying so is
     better than implying it has a witness. One root, at most one producer per
     state, and reachability from that root make the reachable subgraph a
     TREE: a state on a cycle that the root reaches needs an entry edge from
     outside the cycle, and that is a second producer. A cycle can therefore
     only sit in an unreachable component, which the [`Unreachable_state] check
     reports instead.

     The colouring stays anyway, because it is what makes this walk TERMINATE
     rather than what makes it correct. Depending on an earlier check for
     termination is fragile in the one direction that matters: move the
     producer check after this walk and the failure mode is a hung process, not
     a wrong answer. *)
  let colour = Hashtbl.create 16 in
  let out_edges = Hashtbl.create 16 in
  List.iter
    (fun (t : Transition.t) ->
      let prev =
        Option.value
          (Hashtbl.find_opt out_edges t.Transition.before)
          ~default:[]
      in
      Hashtbl.replace out_edges t.Transition.before (t.Transition.after :: prev))
    flow.transitions;
  let rec walk id =
    match Hashtbl.find_opt colour id with
    | Some `Black -> Err.return ()
    | Some `Grey -> Err.fail (`Cycle id)
    | None ->
        Hashtbl.replace colour id `Grey;
        let succs = Option.value (Hashtbl.find_opt out_edges id) ~default:[] in
        let* () = Err.List.iter walk succs in
        Hashtbl.replace colour id `Black;
        Err.return ()
  in
  let* () = walk root.State.id in
  Err.List.iter
    (fun (s : State.t) ->
      if Hashtbl.find_opt colour s.State.id = Some `Black then Err.return ()
      else Err.fail (`Unreachable_state s.State.id))
    flow.states

(* --- the wire ---

   The spine has to reach the browser: [flow_nav.js] maps a selected node id to
   a [Transition.id] to its [Comparison], and [Transition.comparison] exists
   nowhere else. A session that carried the flow only as a rendered graph would
   leave the transition nodes unclickable. *)

let layer_jsont =
  Jsont.enum ~kind:"layer"
    (List.map (fun l -> (Me_ids.Layer.to_string l, l)) Me_ids.Layer.all)

let frame_jsont =
  Jsont.Object.map ~kind:"frame" (fun name iteration ->
      { Pass.Frame.name; iteration })
  |> Jsont.Object.mem "name" Jsont.string ~enc:(fun (f : Pass.Frame.t) ->
      f.Pass.Frame.name)
  |> Jsont.Object.opt_mem "iteration" Jsont.int ~enc:(fun (f : Pass.Frame.t) ->
      f.Pass.Frame.iteration)
  |> Jsont.Object.finish

let exec_id_jsont =
  Jsont.Object.map ~kind:"exec_id" (fun frames leaf index ->
      { Pass.Exec_id.frames; leaf; index })
  |> Jsont.Object.mem "frames" (Jsont.list frame_jsont)
       ~enc:(fun (e : Pass.Exec_id.t) -> e.Pass.Exec_id.frames)
  |> Jsont.Object.mem "leaf" Jsont.string ~enc:(fun (e : Pass.Exec_id.t) ->
      e.Pass.Exec_id.leaf)
  (* [int64_as_string]: the ordinal is [int64] precisely because it may pass
     what a jsoo [int] holds, and the adaptive [Jsont.int64] would emit a JSON
     number that [JSON.parse] rounds. *)
  |> Jsont.Object.mem "index" Jsont.int64_as_string
       ~enc:(fun (e : Pass.Exec_id.t) -> e.Pass.Exec_id.index)
  |> Jsont.Object.finish

let pass_execution_jsont =
  Jsont.Object.map ~kind:"pass_execution" (fun layer exec ->
      { Pass_execution.layer; exec })
  |> Jsont.Object.mem "layer" layer_jsont ~enc:(fun (p : Pass_execution.t) ->
      p.Pass_execution.layer)
  |> Jsont.Object.mem "exec" exec_id_jsont ~enc:(fun (p : Pass_execution.t) ->
      p.Pass_execution.exec)
  |> Jsont.Object.finish

let state_jsont =
  Jsont.Object.map ~kind:"flow_state"
    (fun id graph view layer label produced_by ->
      { State.id; graph; view; layer; label; produced_by })
  |> Jsont.Object.mem "id" Jsont.string ~enc:(fun (s : State.t) -> s.State.id)
  |> Jsont.Object.mem "graph" Jsont.string ~enc:(fun (s : State.t) ->
      s.State.graph)
  |> Jsont.Object.mem "view" Jsont.string ~enc:(fun (s : State.t) ->
      s.State.view)
  |> Jsont.Object.mem "layer" layer_jsont ~enc:(fun (s : State.t) ->
      s.State.layer)
  |> Jsont.Object.mem "label" Jsont.string ~enc:(fun (s : State.t) ->
      s.State.label)
  |> Jsont.Object.opt_mem "producedBy" Jsont.string ~enc:(fun (s : State.t) ->
      s.State.produced_by)
  |> Jsont.Object.finish

(* A tagged union, so a kind that names no execution and a [Pass] that carries
   none are equally unrepresentable on the wire. *)
let kind_jsont =
  Jsont.Object.map ~kind:"transition_kind" (fun kind exec ->
      match (kind, exec) with
      | "import", None -> Transition.Import
      | "pack", None -> Transition.Pack
      | "cross_dialect", None -> Transition.Cross_dialect
      | "adapt", None -> Transition.Adapt
      | "pass", Some e -> Transition.Pass e
      | _ ->
          Jsont.Error.msgf Jsont.Meta.none
            "transition kind %S does not carry its own execution" kind)
  |> Jsont.Object.mem "kind" Jsont.string ~enc:Transition.kind_name
  |> Jsont.Object.opt_mem "exec" pass_execution_jsont ~enc:(function
    | Transition.Pass e -> Some e
    | _ -> None)
  |> Jsont.Object.finish

let transition_jsont =
  Jsont.Object.map ~kind:"flow_transition"
    (fun id before after kind comparison ->
      { Transition.id; before; after; kind; comparison })
  |> Jsont.Object.mem "id" Jsont.string ~enc:(fun (t : Transition.t) ->
      t.Transition.id)
  |> Jsont.Object.mem "before" Jsont.string ~enc:(fun (t : Transition.t) ->
      t.Transition.before)
  |> Jsont.Object.mem "after" Jsont.string ~enc:(fun (t : Transition.t) ->
      t.Transition.after)
  |> Jsont.Object.mem "kind" kind_jsont ~enc:(fun (t : Transition.t) ->
      t.Transition.kind)
  |> Jsont.Object.opt_mem "comparison" Jsont.string
       ~enc:(fun (t : Transition.t) -> t.Transition.comparison)
  |> Jsont.Object.finish

let jsont =
  Jsont.Object.map ~kind:"flow" (fun states transitions graph ->
      { states; transitions; graph })
  |> Jsont.Object.mem "states" (Jsont.list state_jsont) ~enc:(fun t -> t.states)
  |> Jsont.Object.mem "transitions" (Jsont.list transition_jsont) ~enc:(fun t ->
      t.transitions)
  |> Jsont.Object.mem "graph" Jsont.string ~enc:(fun t -> t.graph)
  |> Jsont.Object.finish
