open Graph_ir
open Core.Storage_units

module Mode = struct
  type t = Intermediate | Roles of Storage_script.Config.t

  let equal a b =
    match (a, b) with
    | Intermediate, Intermediate -> true
    | Roles x, Roles y -> Storage_script.Config.equal x y
    | Intermediate, Roles _ | Roles _, Intermediate -> false

  let pp ppf = function
    | Intermediate -> Fmt.string ppf "intermediate"
    | Roles c -> Fmt.pf ppf "roles(%a)" Storage_script.Config.pp c
end

module Pool = struct
  type t = {
    arena : Storage_script.Arena_id.t option;
    kind : Alloc_script.Kind.t;
  }

  let compare a b =
    match Option.compare Storage_script.Arena_id.compare a.arena b.arena with
    | 0 -> Alloc_script.Kind.compare a.kind b.kind
    | c -> c

  let equal a b = compare a b = 0

  let pp ppf p =
    match p.arena with
    | None -> Alloc_script.Kind.pp ppf p.kind
    | Some a ->
        Fmt.pf ppf "%a/%a" Storage_script.Arena_id.pp a Alloc_script.Kind.pp
          p.kind
end

module Limits = struct
  type t = { width : int; expansions : int; state_bytes : Byte_size.t }
  type error = [ `Invalid_limit of [ `Expansions | `State_bytes | `Width ] ]

  let make ~width ~expansions ~state_bytes =
    if width < 1 then Err.fail ~pos:__POS__ (`Invalid_limit `Width)
    else if expansions < 0 then
      Err.fail ~pos:__POS__ (`Invalid_limit `Expansions)
    else Err.return { width; expansions; state_bytes }

  let constructive_only =
    { width = 1; expansions = 0; state_bytes = Byte_size.zero }

  let equal a b =
    a.width = b.width
    && a.expansions = b.expansions
    && Byte_size.equal a.state_bytes b.state_bytes
end

module Config = struct
  type t = {
    mode : Mode.t;
    retain : Release_schedule.Retain.t;
    alignment : Alignment_policy.t;
    limits : Limits.t;
  }
end

module Metrics = struct
  type t = {
    target_peak : Byte_size.t;
    pool_peaks : (Pool.t * Byte_size.t) list;
    pool_peak_sum : Byte_size.t;
    outside_peak : Byte_size.t;
    all_peak : Byte_size.t;
  }

  let equal a b =
    Byte_size.equal a.target_peak b.target_peak
    && List.equal
         (fun (p, x) (q, y) -> Pool.equal p q && Byte_size.equal x y)
         a.pool_peaks b.pool_peaks
    && Byte_size.equal a.pool_peak_sum b.pool_peak_sum
    && Byte_size.equal a.outside_peak b.outside_peak
    && Byte_size.equal a.all_peak b.all_peak

  let pp ppf m =
    Fmt.pf ppf "@[<v>target %a@,pools %a@,pool sum %a@,outside %a@,all %a@]"
      Byte_size.pp m.target_peak
      Fmt.(list ~sep:comma (pair ~sep:(any "=") Pool.pp Byte_size.pp))
      m.pool_peaks Byte_size.pp m.pool_peak_sum Byte_size.pp m.outside_peak
      Byte_size.pp m.all_peak
end

module Stats = struct
  type t = {
    score_evaluations : int;
    expansions : int;
    depth : int;
    max_ready_width : int;
    retained_states : int;
    state_bytes : Byte_size.t;
  }

  let zero =
    {
      score_evaluations = 0;
      expansions = 0;
      depth = 0;
      max_ready_width = 0;
      retained_states = 0;
      state_bytes = Byte_size.zero;
    }
end

module Stop = struct
  type t = Budget_exhausted | Completed | Lower_bound_reached | State_limit
end

module Strategy = struct
  type t = Beam | Identity | Live_first | Peak_first

  let pp ppf t =
    Fmt.string ppf
      (match t with
      | Beam -> "beam"
      | Identity -> "identity"
      | Live_first -> "live-first"
      | Peak_first -> "peak-first")
end

type error =
  [ `Duplicate_node of Node_id.t
  | `Graph of Graph_view.error
  | `Invalid_limit of [ `Expansions | `State_bytes | `Width ]
  | `Not_a_permutation
  | `Structure_changed ]

let pp_error ppf : [< error ] -> unit = function
  | `Duplicate_node id -> Fmt.pf ppf "node %a appears twice" Node_id.pp id
  | `Graph e -> Graph_view.pp_error ppf e
  | `Invalid_limit field ->
      Fmt.pf ppf "invalid scheduler limit: %s"
        (match field with
        | `Expansions -> "expansions"
        | `State_bytes -> "state bytes"
        | `Width -> "width")
  | `Not_a_permutation ->
      Fmt.string ppf "the node lists are not permutations of one another"
  | `Structure_changed ->
      Fmt.string ppf "scheduling changed more than the order of the nodes"

(* Node ids are unique in a validated graph, so sorting by id makes the node
   lists comparable as sets. [Stdlib.compare] is the structural equality of the
   op payloads (NaN equal to itself, as a constant's value should be). *)
let by_id (a : node) (b : node) = Node_id.compare a.Node.id b.Node.id

let same_structure (a : graph) (b : graph) =
  let nodes g = List.sort by_id g.Graph.nodes in
  Stdlib.compare { a with Graph.nodes = [] } { b with Graph.nodes = [] } = 0
  && List.compare_lengths a.Graph.nodes b.Graph.nodes = 0
  && List.for_all2
       (fun (x : node) (y : node) -> Stdlib.compare x y = 0)
       (nodes a) (nodes b)

let check_permutation ~original (g : graph) =
  if not (same_structure original g) then
    Err.fail ~pos:__POS__ `Structure_changed
  else
    Graph_view.of_graph g
    |> Err.map_error ~pos:__POS__ (fun e -> `Graph e)
    |> Err.map (fun _ -> ())

module Result = struct
  type t = {
    graph : graph;
    strategy : Strategy.t;
    stop : Stop.t;
    stats : Stats.t;
  }
end

let identity (g : graph) =
  Graph_view.of_graph g
  |> Err.map_error ~pos:__POS__ (fun e -> `Graph e)
  |> Err.map (fun _ ->
      {
        Result.graph = g;
        strategy = Strategy.Identity;
        stop = Stop.Completed;
        stats = Stats.zero;
      })
