(* See arena_schedule_problem.mli. *)

open Graph_ir
open Core.Storage_units
module Position = Graph_view.Position
module Kind = Alloc_script.Kind
module Metrics = Arena_schedule.Metrics
module Pool = Arena_schedule.Pool

module Block = struct
  type t = {
    id : Tensor_id.t;
    kind : Kind.t;
    bytes : Byte_size.t;
    arena : Storage_script.Arena_id.t option;
    eligible : bool;
    releasable : bool;
  }

  let pool b = { Pool.arena = b.arena; kind = b.kind }
end

type error =
  [ Arena_schedule.error
  | `Arena_script of Tensor_id.t
  | `Dry_run of Eval_direct.error
  | `Order_not_valid of Node_id.t
  | `Peak_bytes_overflow of Tensor_id.t
  | `Too_many_nodes ]

let pp_error ppf : [< error ] -> unit = function
  | #Arena_schedule.error as e -> Arena_schedule.pp_error ppf e
  | `Arena_script id ->
      Fmt.pf ppf "the script allocates or frees %a inconsistently" Tensor_id.pp
        id
  | `Dry_run e -> Eval_direct.pp_error ppf e
  | `Order_not_valid id ->
      Fmt.pf ppf "node %a runs before one of its producers" Node_id.pp id
  | `Peak_bytes_overflow id ->
      Fmt.pf ppf "live bytes overflow at %a" Tensor_id.pp id
  | `Too_many_nodes -> Fmt.string ppf "the graph has too many nodes to schedule"

(* A ceiling on the node count: positions and per-node arrays stay far inside
   js_of_ocaml's 32-bit [int], and so does any product of two of them. *)
let max_nodes = 1_000_000

type t = {
  graph : graph;
  nodes : node array;
  blocks : Block.t list array;
  reads : Block.t list array;
  preds : Position.t list array;
  readers : int Tensor_id.Map.t;
  prefix : Block.t list;
  pools : Pool.t list;  (** the pools with an eligible block, sorted *)
}

let is_sink = function Discard _ -> true | _ -> false
let overflow id (`Quantity_overflow _) = `Peak_bytes_overflow id
let underflow id (`Quantity_underflow _) = `Peak_bytes_overflow id
let add ~id a b = Byte_size.add a b |> Err.map_error ~pos:__POS__ (overflow id)
let sub ~id a b = Byte_size.sub a b |> Err.map_error ~pos:__POS__ (underflow id)
let dedupe ~compare l = List.sort_uniq compare l

(* What a baseline script says, whichever kind it is: every allocation with the
   arena it lives in, the tensors released before the result is published, and
   the allocations that precede the first node. *)
type facts = {
  allocs :
    (Alloc_script.Alloc.t * Storage_script.Arena_id.t option * bool)
    Tensor_id.Map.t;
  freed : Tensor_id.Set.t;
  prefix_ids : Tensor_id.t list;
}

let intermediate_facts script =
  let allocs, freed =
    List.fold_left
      (fun (allocs, freed) -> function
        | Alloc_script.Event.Alloc a ->
            ( Tensor_id.Map.add a.Alloc_script.Alloc.id (a, None, a.eligible)
                allocs,
              freed )
        | Free id -> (allocs, Tensor_id.Set.add id freed)
        | Node _ -> (allocs, freed))
      (Tensor_id.Map.empty, Tensor_id.Set.empty)
      script
  in
  { allocs; freed; prefix_ids = [] }

let role_facts script =
  let execution = function
    | None | Some Storage_script.Arena_id.Constants -> false
    | Some _ -> true
  in
  let allocs, freed, prefix, _, _ =
    List.fold_left
      (fun (allocs, freed, prefix, seen_node, published) -> function
        | Storage_script.Event.Alloc (b : Storage_script.Block.t) ->
            let id = b.alloc.id in
            ( Tensor_id.Map.add id (b.alloc, b.arena, execution b.arena) allocs,
              freed,
              (if seen_node then prefix else id :: prefix),
              seen_node,
              published )
        | Free id when not published ->
            (allocs, Tensor_id.Set.add id freed, prefix, seen_node, published)
        | Free _ -> (allocs, freed, prefix, seen_node, published)
        | Node _ -> (allocs, freed, prefix, true, published)
        | Boundary Result_publication -> (allocs, freed, prefix, seen_node, true)
        | Boundary _ -> (allocs, freed, prefix, seen_node, published))
      (Tensor_id.Map.empty, Tensor_id.Set.empty, [], false, false)
      (Storage_script.events script)
  in
  { allocs; freed; prefix_ids = List.rev prefix }

let of_graph (c : Arena_schedule.Config.t) (g : graph) =
  let open Err.Syntax in
  let* _view =
    Graph_view.of_graph g |> Err.map_error ~pos:__POS__ (fun e -> `Graph e)
  in
  let nodes = Array.of_list g.Graph.nodes in
  let* () =
    if Array.length nodes > max_nodes then Err.fail ~pos:__POS__ `Too_many_nodes
    else Err.return ()
  in
  let* { allocs; freed; prefix_ids } =
    match c.mode with
    | Intermediate ->
        Eval_direct.dry_run ~alignment:c.alignment ~retain:c.retain g
        |> Err.map_error ~pos:__POS__ (fun e -> `Dry_run e)
        |> Err.map intermediate_facts
    | Roles config ->
        Eval_direct.storage_script ~alignment:c.alignment ~retain:c.retain
          config g
        |> Err.map_error ~pos:__POS__ (fun e -> `Dry_run e)
        |> Err.map role_facts
  in
  let block id =
    Tensor_id.Map.find_opt id allocs
    |> Option.map (fun ((a : Alloc_script.Alloc.t), arena, eligible) ->
        {
          Block.id;
          kind = a.kind;
          bytes = a.bytes;
          arena;
          eligible;
          releasable = Tensor_id.Set.mem id freed;
        })
  in
  let producer =
    Array.to_seqi nodes
    |> Seq.fold_left
         (fun m (i, (n : node)) ->
           List.fold_left
             (fun m o -> Tensor_id.Map.add o (Position.of_int i) m)
             m n.Node.outputs)
         Tensor_id.Map.empty
  in
  let blocks =
    Array.map (fun (n : node) -> List.filter_map block n.Node.outputs) nodes
  in
  let operand_ids (n : node) =
    dedupe ~compare:Tensor_id.compare (Graph_ir.operands n.Node.op)
  in
  let reads =
    Array.map
      (fun (n : node) ->
        if is_sink n.Node.op then [] else List.filter_map block (operand_ids n))
      nodes
  in
  let preds =
    Array.map
      (fun n ->
        List.filter_map
          (fun o -> Tensor_id.Map.find_opt o producer)
          (operand_ids n)
        |> dedupe ~compare:Position.compare)
      nodes
  in
  let readers =
    Array.fold_left
      (fun m rs ->
        List.fold_left
          (fun m (b : Block.t) ->
            Tensor_id.Map.update b.id
              (fun n -> Some (1 + Option.value n ~default:0))
              m)
          m rs)
      Tensor_id.Map.empty reads
  in
  let prefix = List.filter_map block prefix_ids in
  let pools =
    let eligible =
      List.filter (fun (b : Block.t) -> b.eligible) prefix
      @ List.concat (Array.to_list blocks)
      |> List.filter (fun (b : Block.t) -> b.eligible)
    in
    List.sort_uniq Pool.compare (List.map Block.pool eligible)
  in
  Err.return { graph = g; nodes; blocks; reads; preds; readers; prefix; pools }

let graph t = t.graph
let prefix t = t.prefix
let node_count t = Array.length t.nodes
let node t p = t.nodes.((p : Position.t :> int))
let blocks t p = t.blocks.((p : Position.t :> int))
let reads t p = t.reads.((p : Position.t :> int))
let preds t p = t.preds.((p : Position.t :> int))
let readers t id = Option.value (Tensor_id.Map.find_opt id t.readers) ~default:0

let is_valid_order t order =
  let n = node_count t in
  if Array.length order <> n then Err.fail ~pos:__POS__ `Not_a_permutation
  else
    let seen = Array.make n false in
    let rec go i =
      if i = n then Err.return ()
      else
        let p = order.(i) in
        let k = (p : Position.t :> int) in
        if k < 0 || k >= n || seen.(k) then
          Err.fail ~pos:__POS__ `Not_a_permutation
        else if
          not
            (List.for_all (fun q -> seen.((q : Position.t :> int))) (preds t p))
        then Err.fail ~pos:__POS__ (`Order_not_valid (node t p).Node.id)
        else (
          seen.(k) <- true;
          go (i + 1))
    in
    go 0

let reorder t order =
  let open Err.Syntax in
  let* () = is_valid_order t order in
  Err.return
    { t.graph with Graph.nodes = Array.to_list (Array.map (node t) order) }

(* Live-byte accounting shared by the replay and the bound: totals plus one
   running figure per pool, each with the largest value it has held. *)
module Account = struct
  type t = {
    pools : Pool.t list;
    live : Byte_size.t array;  (** all; target; outside; then one per pool *)
    peak : Byte_size.t array;
  }

  let all = 0
  let target = 1
  let outside = 2

  let create pools =
    let n = 3 + List.length pools in
    {
      pools;
      live = Array.make n Byte_size.zero;
      peak = Array.make n Byte_size.zero;
    }

  let slot t (b : Block.t) =
    let want = Block.pool b in
    let rec index i = function
      | [] -> None
      | p :: rest -> if Pool.equal p want then Some i else index (i + 1) rest
    in
    index 3 t.pools

  (* A block counts toward the total always, toward the target and its pool
     when eligible, and toward the outside figure when no arena holds it. *)
  let cells t (b : Block.t) =
    if b.eligible then
      all :: target :: (match slot t b with Some i -> [ i ] | None -> [])
    else if Option.is_none b.arena then [ all; outside ]
    else [ all ]

  let alloc t (b : Block.t) =
    let open Err.Syntax in
    List.fold_left
      (fun acc i ->
        let* () = acc in
        let* v = add ~id:b.id t.live.(i) b.bytes in
        t.live.(i) <- v;
        t.peak.(i) <- Byte_size.max t.peak.(i) v;
        Err.return ())
      (Err.return ()) (cells t b)

  let free t (b : Block.t) =
    let open Err.Syntax in
    List.fold_left
      (fun acc i ->
        let* () = acc in
        let* v = sub ~id:b.id t.live.(i) b.bytes in
        t.live.(i) <- v;
        Err.return ())
      (Err.return ()) (cells t b)

  let metrics t =
    let open Err.Syntax in
    let pools = List.mapi (fun i p -> (p, t.peak.(3 + i))) t.pools in
    let* sum =
      List.fold_left
        (fun acc (_, b) ->
          let* s = acc in
          add ~id:(Tensor_id.of_int 0) s b)
        (Err.return Byte_size.zero)
        pools
    in
    Err.return
      {
        Metrics.target_peak = t.peak.(target);
        pool_peaks = pools;
        pool_peak_sum = sum;
        outside_peak = t.peak.(outside);
        all_peak = t.peak.(all);
      }
end

let iter_result f l =
  List.fold_left (fun acc x -> Err.bind acc (fun () -> f x)) (Err.return ()) l

let metrics t order =
  let open Err.Syntax in
  let* () = is_valid_order t order in
  let acct = Account.create t.pools in
  let remaining = Hashtbl.create 64 in
  let left (b : Block.t) =
    match Hashtbl.find_opt remaining b.id with
    | Some n -> n
    | None -> readers t b.id
  in
  let step p =
    let outs = blocks t p in
    let* () = iter_result (Account.alloc acct) outs in
    let* () =
      iter_result
        (fun (b : Block.t) ->
          let n = left b - 1 in
          Hashtbl.replace remaining b.id n;
          if n = 0 && b.releasable then Account.free acct b else Err.return ())
        (reads t p)
    in
    iter_result
      (fun (b : Block.t) ->
        if readers t b.id = 0 && b.releasable then Account.free acct b
        else Err.return ())
      outs
  in
  let* () = iter_result (Account.alloc acct) t.prefix in
  let* () =
    iter_result
      (fun (b : Block.t) ->
        if readers t b.id = 0 && b.releasable then Account.free acct b
        else Err.return ())
      t.prefix
  in
  let* () = iter_result step (Array.to_list order) in
  Account.metrics acct

let lower_bound t =
  let open Err.Syntax in
  let acct = Account.create t.pools in
  let raise_to one =
    Array.iteri
      (fun j v -> acct.peak.(j) <- Byte_size.max acct.peak.(j) v)
      one.Account.peak
  in
  (* The fixed prefix is allocated whatever the order. *)
  let* () =
    let one = Account.create t.pools in
    let* () = iter_result (Account.alloc one) t.prefix in
    raise_to one;
    Err.return ()
  in
  let step i (n : node) =
    if is_sink n.Node.op then Err.return ()
    else
      let p = Position.of_int i in
      (* What the node needs while it runs, alone: its outputs and operands. *)
      let one = Account.create t.pools in
      let* () = iter_result (Account.alloc one) (blocks t p @ reads t p) in
      raise_to one;
      Err.return ()
  in
  let* () =
    Array.to_seqi t.nodes |> List.of_seq |> iter_result (fun (i, n) -> step i n)
  in
  Account.metrics acct

(* The independent path. *)
let fresh_intermediate (c : Arena_schedule.Config.t) (g : graph) =
  let open Err.Syntax in
  let* script =
    Eval_direct.dry_run ~alignment:c.alignment ~retain:c.retain g
    |> Err.map_error ~pos:__POS__ (fun e -> `Dry_run e)
  in
  let* all_peak =
    Alloc_script.peak_bytes script
    |> Err.map_error ~pos:__POS__ (fun (`Peak_bytes_overflow id) ->
        `Peak_bytes_overflow id)
  in
  let rec fold live_t peak_t live_o peak_o sizes = function
    | [] -> Err.return (peak_t, peak_o)
    | Alloc_script.Event.Alloc (a : Alloc_script.Alloc.t) :: rest ->
        let sizes = Tensor_id.Map.add a.id a sizes in
        if a.eligible then
          let* live_t = add ~id:a.id live_t a.bytes in
          fold live_t (Byte_size.max peak_t live_t) live_o peak_o sizes rest
        else
          let* live_o = add ~id:a.id live_o a.bytes in
          fold live_t peak_t live_o (Byte_size.max peak_o live_o) sizes rest
    | Free id :: rest -> (
        match Tensor_id.Map.find_opt id sizes with
        | None -> Err.fail ~pos:__POS__ (`Arena_script id)
        | Some a ->
            if a.eligible then
              let* live_t = sub ~id live_t a.bytes in
              fold live_t peak_t live_o peak_o sizes rest
            else
              let* live_o = sub ~id live_o a.bytes in
              fold live_t peak_t live_o peak_o sizes rest)
    | Node _ :: rest -> fold live_t peak_t live_o peak_o sizes rest
  in
  let z = Byte_size.zero in
  let* target_peak, outside_peak = fold z z z z Tensor_id.Map.empty script in
  let* problem =
    Arena_problem.of_script script
    |> Err.map_error ~pos:__POS__ (fun (`Arena_script id) -> `Arena_script id)
  in
  let* pools =
    List.fold_left
      (fun acc (kp : Arena_problem.Kind_problem.t) ->
        let* acc = acc in
        let* b =
          Interval_alloc.lower_bound kp.script
          |> Err.map_error ~pos:__POS__ (fun (`Live_overflow id) ->
              `Peak_bytes_overflow id)
        in
        Err.return (({ Pool.arena = None; kind = kp.kind }, b) :: acc))
      (Err.return [])
      (Arena_problem.kinds problem)
  in
  let pools = List.rev pools in
  let* pool_peak_sum =
    List.fold_left
      (fun acc (_, b) ->
        let* s = acc in
        add ~id:(Tensor_id.of_int 0) s b)
      (Err.return z) pools
  in
  Err.return
    {
      Metrics.target_peak;
      pool_peaks = pools;
      pool_peak_sum;
      outside_peak;
      all_peak;
    }

let fresh_roles (c : Arena_schedule.Config.t) config (g : graph) =
  let open Err.Syntax in
  let* script =
    Eval_direct.storage_script ~alignment:c.alignment ~retain:c.retain config g
    |> Err.map_error ~pos:__POS__ (fun e -> `Dry_run e)
  in
  let peak where =
    Storage_script.peak_bytes script ~where
    |> Err.map_error ~pos:__POS__ (fun (`Peak_bytes_overflow id) ->
        `Peak_bytes_overflow id)
  in
  let execution (b : Storage_script.Block.t) =
    match b.arena with
    | None | Some Storage_script.Arena_id.Constants -> false
    | Some _ -> true
  in
  let pools =
    List.filter_map
      (function
        | Storage_script.Event.Alloc (b : Storage_script.Block.t)
          when execution b ->
            Some { Pool.arena = b.arena; kind = b.alloc.kind }
        | _ -> None)
      (Storage_script.events script)
    |> List.sort_uniq Pool.compare
  in
  let* target_peak = peak execution in
  let* outside_peak =
    peak (fun (b : Storage_script.Block.t) -> Option.is_none b.arena)
  in
  let* all_peak = peak (fun _ -> true) in
  let* pool_peaks =
    List.fold_left
      (fun acc (pool : Pool.t) ->
        let* acc = acc in
        let* v =
          peak (fun (b : Storage_script.Block.t) ->
              b.arena = pool.arena && Kind.equal b.alloc.kind pool.kind)
        in
        Err.return ((pool, v) :: acc))
      (Err.return []) pools
  in
  let pool_peaks = List.rev pool_peaks in
  let* pool_peak_sum =
    List.fold_left
      (fun acc (_, b) ->
        let* s = acc in
        add ~id:(Tensor_id.of_int 0) s b)
      (Err.return Byte_size.zero)
      pool_peaks
  in
  Err.return
    { Metrics.target_peak; pool_peaks; pool_peak_sum; outside_peak; all_peak }

let fresh_metrics (c : Arena_schedule.Config.t) g =
  match c.mode with
  | Intermediate -> fresh_intermediate c g
  | Roles config -> fresh_roles c config g
