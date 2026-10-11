(* A transformation pass's correspondence (clusters of nodes and of tensors
   between two graph versions) written as a mapping artifact between their
   descriptions. The clusters say which nodes became which, not where a legal
   execution boundary lies, so the provider's job is to coarsen: nodes of a
   cluster form one unit, the producers of a value cluster's tensors share a
   unit, and a value that only one side exposes at a boundary is made internal
   by merging its producers with its consumers. Nothing is discovered by
   comparing values: the clusters and their claims are the pass's own. The
   result is still a candidate for the planner. *)

open Inference_compare_schema
module Tid = Graph_ir.Tensor_id
module Nid = Graph_ir.Node_id
module G = Graph_model

let ( let* ) = Result.bind
let text = G.text
let all = G.all

type 'op side = {
  model : 'op G.t;
  bytes : Tid.t -> (string, string) result;
      (** The exact value of a source tensor of this graph. *)
}

type t = {
  mapping : Mapping_artifact.t;
  pins : (Id.Canonical_value.t * (unit -> (string, string) result)) list;
      (** The value of each canonical value some graph source holds, read on
          demand: a composition keeps only those the end graphs cannot supply
          themselves (a created or folded constant). *)
}

module Uf = struct
  type t = (string, string) Hashtbl.t

  let create () : t = Hashtbl.create 64

  let rec find (t : t) x =
    match Hashtbl.find_opt t x with
    | None ->
        Hashtbl.replace t x x;
        x
    | Some p when String.equal p x -> x
    | Some p ->
        let r = find t p in
        Hashtbl.replace t x r;
        r

  let union t a b =
    let ra = find t a and rb = find t b in
    if not (String.equal ra rb) then Hashtbl.replace t ra rb;
    not (String.equal ra rb)
end

(* One side of the pass: the graph with its dataflow indexed. *)
type 'op index = {
  m : 'op G.t;
  prefix : string;
  producer : (int, Nid.t) Hashtbl.t;
  consumers : (int, Nid.t list) Hashtbl.t;
  node : (int, 'op Graph_common.Node.t) Hashtbl.t;
}

let index (type op) prefix (m : op G.t) : op index =
  let producer = Hashtbl.create 64
  and consumers = Hashtbl.create 64
  and node = Hashtbl.create 64 in
  List.iter
    (fun (n : op Graph_common.Node.t) ->
      Hashtbl.replace node (Nid.to_int n.id) n;
      List.iter
        (fun t -> Hashtbl.replace producer (Tid.to_int t) n.id)
        n.outputs;
      List.iter
        (fun t ->
          let k = Tid.to_int t in
          Hashtbl.replace consumers k
            (n.id :: Option.value ~default:[] (Hashtbl.find_opt consumers k)))
        (m.ops.operands n.op))
    m.graph.nodes;
  { m; prefix; producer; consumers; node }

let key_of ix n = ix.prefix ^ string_of_int (Nid.to_int n)

let consumers_of ix t =
  Option.value ~default:[] (Hashtbl.find_opt ix.consumers (Tid.to_int t))

let is_final ix t = List.exists (Tid.equal t) ix.m.graph.outputs

(* What the nodes of one block read from outside it and expose to the outside
   or to the graph's result list, in ascending tensor order. *)
let boundary ix ~inside (ns : Nid.t list) =
  let mine n = inside (key_of ix n) in
  let nodes = List.map (fun n -> Hashtbl.find ix.node (Nid.to_int n)) ns in
  let ins =
    List.concat_map
      (fun (n : _ Graph_common.Node.t) ->
        List.filter
          (fun t ->
            match Hashtbl.find_opt ix.producer (Tid.to_int t) with
            | Some p -> not (mine p)
            | None -> true)
          (ix.m.ops.operands n.op))
      nodes
  and outs =
    List.concat_map
      (fun (n : _ Graph_common.Node.t) ->
        List.filter
          (fun t ->
            is_final ix t
            || List.exists (fun c -> not (mine c)) (consumers_of ix t))
          n.outputs)
      nodes
  in
  (List.sort_uniq Tid.compare ins, List.sort_uniq Tid.compare outs)

type block = { s : Nid.t list; d : Nid.t list }

let claim_of (r : Correspondence.relation) : Claim.t =
  match r with
  | Identical -> Identical
  | Equivalent -> Equivalent
  | Unverifiable -> Unverifiable
  | Approximate set ->
      Approximate
        (List.filter_map
           (fun (p : Correspondence.Precision.t) ->
             let (Payload.Fmt f) = p.fmt in
             match Id.Precision.of_string (Payload.fmt_name f) with
             | Ok id -> Some id
             | Error _ -> None)
           (Correspondence.Precision.Set.elements set))

(* The steps that read a value of spec [from] as one of spec [into]. *)
let steps ~(from : Tensor_spec.t) ~(into : Tensor_spec.t) =
  if Tensor_spec.equal from into then Ok []
  else if
    Tensor_spec.equal { from with logical_shape = into.logical_shape } into
  then Ok [ Conversion_step.Reshape into.logical_shape ]
  else Error "the two specs differ by more than a shape"

(* A recipe applying [steps] in order to [arg]. *)
let chain ~id arg steps (output_spec : Tensor_spec.t) =
  let* id = text (Id.Recipe.of_string id) in
  let rec build i prev acc = function
    | [] -> Ok (List.rev acc, prev)
    | step :: rest ->
        let* nid =
          text (Id.Conversion_node.of_string (Printf.sprintf "s%d" i))
        in
        build (i + 1) (Recipe.Conversion_result nid)
          ({ Recipe.id = nid; operation = step; arguments = [ prev ] } :: acc)
          rest
  in
  let* nodes, result = build 0 arg [] steps in
  Ok { Recipe.id; nodes; result; output_spec; lossless = true }

let make (type a b) ~name ~digest ~(src : a side) ~(dst : b side)
    ~(export : Map_export.t) : (t, string) result =
  let sx = index "s" src.model and dx = index "d" dst.model in
  let vcs = Array.of_list export.values in
  let cs = Hashtbl.create 64 and cd = Hashtbl.create 64 in
  Array.iteri
    (fun i (c : Map_export.Value_cluster.t) ->
      List.iter (fun t -> Hashtbl.replace cs (Tid.to_int t) i) c.src;
      List.iter (fun t -> Hashtbl.replace cd (Tid.to_int t) i) c.dst)
    vcs;
  let key_s t = Hashtbl.find_opt cs (Tid.to_int t)
  and key_d t = Hashtbl.find_opt cd (Tid.to_int t) in
  (* ---- blocks ----------------------------------------------------------- *)
  let uf = Uf.create () in
  List.iter
    (fun (n : a Graph_common.Node.t) -> ignore (Uf.find uf (key_of sx n.id)))
    src.model.graph.nodes;
  List.iter
    (fun (n : b Graph_common.Node.t) -> ignore (Uf.find uf (key_of dx n.id)))
    dst.model.graph.nodes;
  let union_all = function
    | [] -> false
    | x :: rest -> List.fold_left (fun ch y -> Uf.union uf x y || ch) false rest
  in
  let producers_of_cluster (c : Map_export.Value_cluster.t) =
    List.filter_map
      (fun t ->
        Option.map (key_of sx) (Hashtbl.find_opt sx.producer (Tid.to_int t)))
      c.src
    @ List.filter_map
        (fun t ->
          Option.map (key_of dx) (Hashtbl.find_opt dx.producer (Tid.to_int t)))
        c.dst
  in
  let consumers_of_cluster (c : Map_export.Value_cluster.t) =
    List.concat_map (fun t -> List.map (key_of sx) (consumers_of sx t)) c.src
    @ List.concat_map (fun t -> List.map (key_of dx) (consumers_of dx t)) c.dst
  in
  List.iter
    (fun (c : Map_export.Node_cluster.t) ->
      ignore
        (union_all (List.map (key_of sx) c.src @ List.map (key_of dx) c.dst)))
    export.nodes;
  Array.iter (fun c -> ignore (union_all (producers_of_cluster c))) vcs;
  let blocks () =
    let tbl = Hashtbl.create 64 in
    let upd root f =
      let b =
        Option.value ~default:{ s = []; d = [] } (Hashtbl.find_opt tbl root)
      in
      Hashtbl.replace tbl root (f b)
    in
    List.iter
      (fun (n : a Graph_common.Node.t) ->
        upd (Uf.find uf (key_of sx n.id)) (fun b -> { b with s = n.id :: b.s }))
      src.model.graph.nodes;
    List.iter
      (fun (n : b Graph_common.Node.t) ->
        upd (Uf.find uf (key_of dx n.id)) (fun b -> { b with d = n.id :: b.d }))
      dst.model.graph.nodes;
    let l =
      Hashtbl.fold
        (fun root b acc ->
          ( root,
            { s = List.sort Nid.compare b.s; d = List.sort Nid.compare b.d } )
          :: acc)
        tbl []
    in
    let first (_, b) =
      match (b.s, b.d) with
      | n :: _, _ -> (0, Nid.to_int n)
      | [], n :: _ -> (1, Nid.to_int n)
      | [], [] -> (2, 0)
    in
    List.sort (fun x y -> compare (first x) (first y)) l
  in
  let dup l =
    let rec go seen = function
      | [] -> []
      | x :: rest ->
          if List.mem x seen then x :: go seen rest else go (x :: seen) rest
    in
    go [] l
  in
  let keys_of key l = List.filter_map key l in
  let unit_boundaries (root, b) =
    let inside k = String.equal (Uf.find uf k) root in
    let s_in, s_out = boundary sx ~inside b.s
    and d_in, d_out = boundary dx ~inside b.d in
    (s_in, s_out, d_in, d_out)
  in
  (* Units depend on each other through either side's dataflow; a strongly
     connected set of blocks has no execution order, so it becomes one. *)
  let merge_cycles () =
    let edges = Hashtbl.create 64 in
    let add a b =
      if not (String.equal a b) then
        Hashtbl.replace edges a
          (b :: Option.value ~default:[] (Hashtbl.find_opt edges a))
    in
    let side ix =
      Hashtbl.iter
        (fun _ (n : _ Graph_common.Node.t) ->
          List.iter
            (fun t ->
              match Hashtbl.find_opt ix.producer (Tid.to_int t) with
              | Some p ->
                  add (Uf.find uf (key_of ix p)) (Uf.find uf (key_of ix n.id))
              | None -> ())
            (ix.m.ops.operands n.op))
        ix.node
    in
    side sx;
    side dx;
    let index = Hashtbl.create 64 and low = Hashtbl.create 64 in
    let on_stack = Hashtbl.create 64 and stack = ref [] and counter = ref 0 in
    let merged = ref false in
    let rec visit v =
      Hashtbl.replace index v !counter;
      Hashtbl.replace low v !counter;
      incr counter;
      stack := v :: !stack;
      Hashtbl.replace on_stack v ();
      List.iter
        (fun w ->
          if not (Hashtbl.mem index w) then (
            visit w;
            Hashtbl.replace low v
              (min (Hashtbl.find low v) (Hashtbl.find low w)))
          else if Hashtbl.mem on_stack w then
            Hashtbl.replace low v
              (min (Hashtbl.find low v) (Hashtbl.find index w)))
        (Option.value ~default:[] (Hashtbl.find_opt edges v));
      if Hashtbl.find low v = Hashtbl.find index v then
        let rec pop acc =
          match !stack with
          | w :: rest ->
              stack := rest;
              Hashtbl.remove on_stack w;
              if String.equal w v then w :: acc else pop (w :: acc)
          | [] -> acc
        in
        let comp = pop [] in
        if List.length comp > 1 && union_all comp then merged := true
    in
    let roots = Hashtbl.fold (fun k _ acc -> k :: acc) edges [] in
    List.iter (fun v -> if not (Hashtbl.mem index v) then visit v) roots;
    !merged
  in
  let rec settle n =
    let bl = blocks () in
    let changed = ref false in
    List.iter
      (fun blk ->
        let _, s_out, _, d_out = unit_boundaries blk in
        let ks = keys_of key_s s_out and kd = keys_of key_d d_out in
        let missing a b = List.filter (fun k -> not (List.mem k b)) a in
        let bad =
          List.sort_uniq compare
            (dup ks @ dup kd @ missing ks kd @ missing kd ks)
        in
        List.iter
          (fun k ->
            let c = vcs.(k) in
            if union_all (producers_of_cluster c @ consumers_of_cluster c) then
              changed := true)
          bad)
      bl;
    if (not !changed) && merge_cycles () then changed := true;
    if !changed && n < 256 then settle (n + 1) else bl
  in
  let bl = settle 0 in
  (* ---- the artifact ----------------------------------------------------- *)
  let skey = src.model.key and dkey = dst.model.key in
  let tspec_s t = G.spec_of src.model.graph t
  and tspec_d t = G.spec_of dst.model.graph t in
  let sref t = G.tensor_ref ~key:skey ~scope:[] t
  and dref t = G.tensor_ref ~key:dkey ~scope:[] t in
  let cname k = Printf.sprintf "%sv%05d" name k in
  let ckey k = text (Id.Canonical_value.of_string (cname k)) in
  let canon_spec k =
    let c = vcs.(k) in
    match (c.src, c.dst) with
    | t :: _, _ -> tspec_s t
    | [], t :: _ -> tspec_d t
    | [], [] -> Error "empty value cluster"
  in
  let unit_name i = Printf.sprintf "%su%05d" name i in
  let side_tag = Side.to_string in
  let infos =
    List.mapi (fun i ((_, blk) as x) -> (i, blk, unit_boundaries x)) bl
  in
  (* every unit exposes the same values on both sides, once each *)
  let* () =
    let rec go = function
      | [] -> Ok ()
      | (i, _, (_, s_out, _, d_out)) :: rest ->
          let ks = keys_of key_s s_out and kd = keys_of key_d d_out in
          if
            List.length ks <> List.length s_out
            || List.length kd <> List.length d_out
          then
            Error (Printf.sprintf "unit %d: an output has no correspondence" i)
          else if dup ks <> [] || dup kd <> [] then
            Error (Printf.sprintf "unit %d: two outputs stand for one value" i)
          else if List.sort compare ks <> List.sort compare kd then
            Error
              (Printf.sprintf "unit %d: outputs differ: left [%s] right [%s]" i
                 (String.concat "," (List.map cname ks))
                 (String.concat "," (List.map cname kd)))
          else go rest
    in
    go infos
  in
  let key_of_in side t =
    match if side = Side.Left then key_s t else key_d t with
    | Some k -> Ok k
    | None -> Error "an input has no correspondence"
  in
  let* units =
    all
      (fun (i, blk, (s_in, s_out, d_in, d_out)) ->
        let* unit_id = text (Id.Unit.of_string (unit_name i)) in
        let* lnodes = all (G.node_ref ~key:skey ~scope:[]) blk.s in
        let* rnodes = all (G.node_ref ~key:dkey ~scope:[]) blk.d in
        let* lin = all sref s_in in
        let* lout = all sref s_out in
        let* rin = all dref d_in in
        let* rout = all dref d_out in
        Ok
          {
            Matched_unit.unit_id;
            left = { Cluster.nodes = lnodes; inputs = lin; outputs = lout };
            right = { Cluster.nodes = rnodes; inputs = rin; outputs = rout };
            output_pairs = [];
          })
      infos
  in
  let produced =
    List.concat_map (fun (_, _, (_, s_out, _, _)) -> keys_of key_s s_out) infos
  in
  let* input_bindings, input_checks, output_bindings, read_keys =
    let bs = ref [] and cks = ref [] and obs = ref [] and reads = ref [] in
    let* () =
      let rec go = function
        | [] -> Ok ()
        | (i, _, (s_in, s_out, d_in, d_out)) :: rest ->
            let uid = unit_name i in
            let* unit = text (Id.Unit.of_string uid) in
            let side_inputs side ins tspec tref =
              let rec each seen = function
                | [] -> Ok ()
                | t :: more ->
                    let* k = key_of_in side t in
                    let* ckey_ = ckey k in
                    let* canonical = canon_spec k in
                    let* local = tspec t in
                    let* tr = tref t in
                    let* to_side = steps ~from:canonical ~into:local in
                    let* from_side = steps ~from:local ~into:canonical in
                    let tag = G.tensor_local t in
                    let* b =
                      chain
                        ~id:
                          (Printf.sprintf "b-%s-%s-%s" uid (side_tag side) tag)
                        (Recipe.Canonical ckey_) to_side local
                    in
                    bs :=
                      !bs
                      @ [
                          {
                            Input_binding.unit;
                            side;
                            tensor = tr;
                            from_common = b;
                          };
                        ];
                    reads := k :: !reads;
                    if List.mem k seen then each seen more
                    else
                      let* c =
                        chain
                          ~id:
                            (Printf.sprintf "k-%s-%s-%s" uid (side_tag side) tag)
                          (Recipe.Boundary tr) from_side canonical
                      in
                      cks :=
                        !cks
                        @ [
                            {
                              Input_check.unit;
                              side;
                              canonical = ckey_;
                              from_imported_inputs = c;
                            };
                          ];
                      each (k :: seen) more
              in
              each [] ins
            in
            let* () = side_inputs Left s_in tspec_s sref in
            let* () = side_inputs Right d_in tspec_d dref in
            let* () =
              all
                (fun lt ->
                  let k = Option.get (key_s lt) in
                  let rt = List.find (fun t -> key_d t = Some k) d_out in
                  let* ckey_ = ckey k in
                  let* canonical = canon_spec k in
                  let* lspec = tspec_s lt in
                  let* rspec = tspec_d rt in
                  let* ls = steps ~from:lspec ~into:canonical in
                  let* rs = steps ~from:rspec ~into:canonical in
                  let* lr = sref lt in
                  let* rr = dref rt in
                  let* l =
                    chain
                      ~id:(Printf.sprintf "o-%s-left-%s" uid (cname k))
                      (Recipe.Boundary lr) ls canonical
                  in
                  let* r =
                    chain
                      ~id:(Printf.sprintf "o-%s-right-%s" uid (cname k))
                      (Recipe.Boundary rr) rs canonical
                  in
                  let* comparison_id =
                    text (Id.Comparison.of_string ("cmp:" ^ cname k))
                  in
                  obs :=
                    !obs
                    @ [
                        {
                          Output_binding.unit;
                          key = ckey_;
                          from_left_outputs = l;
                          from_right_outputs = r;
                          comparison_id;
                        };
                      ];
                  Ok ())
                s_out
              |> Result.map ignore
            in
            go rest
      in
      go infos
    in
    Ok (!bs, !cks, !obs, !reads)
  in
  (* final outputs, by position *)
  let* final_outputs =
    let l = src.model.graph.outputs and r = dst.model.graph.outputs in
    if List.length l <> List.length r then
      Error "the two graphs return a different number of outputs"
    else
      all
        (fun (i, (lt, rt)) ->
          let* occurrence =
            text (Id.Output_occurrence.of_string (Printf.sprintf "out%d" i))
          in
          match (key_s lt, key_d rt) with
          | Some a, Some b when a = b ->
              let* canonical = ckey a in
              let* left = sref lt in
              let* right = dref rt in
              Ok { Final_output.occurrence; canonical; left; right }
          | _ ->
              Error
                (Printf.sprintf
                   "output %d: the two graphs return different values" i))
        (List.mapi (fun i x -> (i, x)) (List.combine l r))
  in
  let used_keys =
    List.sort_uniq compare
      (read_keys @ produced
      @ List.filter_map
          (fun (f : Final_output.t) ->
            let s = Id.Canonical_value.to_string f.canonical in
            let n = String.length name in
            int_of_string_opt (String.sub s (n + 1) (String.length s - n - 1)))
          final_outputs)
  in
  let* canonical_values =
    all
      (fun k ->
        let* key = ckey k in
        let* spec = canon_spec k in
        Ok { Canonical_spec.key; spec })
      used_keys
  in
  (* a value nothing produces is a source: pinned by its payload *)
  let* sources =
    all
      (fun k ->
        let c = vcs.(k) in
        let is_source ix t = List.exists (Tid.equal t) ix.m.graph.inputs in
        let* key = ckey k in
        let read =
          match
            ( List.find_opt (is_source sx) c.src,
              List.find_opt (is_source dx) c.dst )
          with
          | Some t, _ -> Ok (fun () -> src.bytes t)
          | None, Some t -> Ok (fun () -> dst.bytes t)
          | None, None ->
              Error (cname k ^ ": nothing produces or pins this value")
        in
        let* read = read in
        Ok (key, read))
      (List.filter (fun k -> not (List.mem k produced)) used_keys)
  in
  let* initial_values =
    all
      (fun (key, read) ->
        let* bytes = read () in
        let* payload_digest = digest bytes in
        let* payload_id =
          text
            (Id.Payload.of_string ("pass/" ^ Id.Canonical_value.to_string key))
        in
        Ok { Initial_binding.key; payload_id; payload_digest })
      sources
  in
  let reason = text (Prose.Description.of_string "by the pass") in
  let* dispositions =
    all Fun.id
      (List.concat_map
         (fun (c : Map_export.Node_cluster.t) ->
           if c.dst = [] then
             List.map
               (fun n ->
                 let* node = G.node_ref ~key:skey ~scope:[] n in
                 let* reason = reason in
                 Ok (Member_disposition.Eliminated { node; reason }))
               c.src
           else if c.src = [] then
             List.map
               (fun n ->
                 let* node = G.node_ref ~key:dkey ~scope:[] n in
                 let* reason = reason in
                 Ok (Member_disposition.Created { node; reason }))
               c.dst
           else [])
         export.nodes)
  in
  let* tensor_relations =
    all
      (fun k ->
        let c = vcs.(k) in
        let* relation_id =
          text (Id.Tensor_relation.of_string ("rel:" ^ cname k))
        in
        let* left_members = all sref c.src in
        let* right_members = all dref c.dst in
        let route dir ~src_ref ~dst_ref ~from ~into tag =
          let* route_id =
            text
              (Id.Value_route.of_string
                 (Printf.sprintf "%s:%s:%s" dir (cname k) tag))
          in
          let* st = steps ~from ~into in
          Ok
            {
              Value_route.route_id;
              sources = [ src_ref ];
              destination = dst_ref;
              steps = st;
              claim = claim_of c.claim;
              lossy = false;
            }
        in
        let* routes =
          all Fun.id
            (List.concat_map
               (fun l ->
                 List.map
                   (fun r ->
                     let tag = G.tensor_local l ^ G.tensor_local r in
                     let* lr = sref l in
                     let* rr = dref r in
                     let* ls = tspec_s l in
                     let* rs = tspec_d r in
                     let* f =
                       route "to" ~src_ref:lr ~dst_ref:rr ~from:ls ~into:rs tag
                     in
                     let* b =
                       route "from" ~src_ref:rr ~dst_ref:lr ~from:rs ~into:ls
                         tag
                     in
                     Ok [ f; b ])
                   c.dst)
               c.src)
          |> Result.map List.concat
        in
        Ok
          {
            Tensor_relation.relation_id;
            left_members;
            right_members;
            member_pairs = [];
            routes;
          })
      (List.filter
         (fun k -> vcs.(k).src <> [] && vcs.(k).dst <> [])
         (List.init (Array.length vcs) Fun.id))
  in
  Ok
    {
      pins = sources;
      mapping =
        {
          Mapping_artifact.schema_version = V1;
          left_graph = skey;
          right_graph = dkey;
          units;
          tensor_relations;
          dispositions;
          frontier =
            {
              Frontier_mapping.canonical_values;
              final_outputs;
              initial_values;
              input_bindings;
              input_checks;
              output_bindings;
            };
          derivation_digest = [];
        };
    }

(* The correspondence of a graph with itself: every node and every tensor is
   its own cluster. *)
let identity_export (g : _ Graph_common.Graph.t) : Map_export.t =
  {
    nodes =
      List.map
        (fun (n : _ Graph_common.Node.t) ->
          { Map_export.Node_cluster.src = [ n.id ]; dst = [ n.id ] })
        g.nodes;
    values =
      List.map
        (fun (id, _) ->
          {
            Map_export.Value_cluster.src = [ id ];
            dst = [ id ];
            claim = Identical;
          })
        (Tid.Map.bindings g.tensors);
  }

(* The same clusters read from the other graph to this one. A claim is
   symmetric, so only the two sides change places. *)
let reverse_export (e : Map_export.t) : Map_export.t =
  {
    nodes =
      List.map
        (fun (c : Map_export.Node_cluster.t) ->
          { Map_export.Node_cluster.src = c.dst; dst = c.src })
        e.nodes;
    values =
      List.map
        (fun (c : Map_export.Value_cluster.t) ->
          { c with Map_export.Value_cluster.src = c.dst; dst = c.src })
        e.values;
  }

(* The pinned values of [pins] the mapping's frontier names and [known] does
   not already hold, as exact logical values. *)
let pinned_values ~ctx ~(mapping : Mapping_artifact.t) ~known pins =
  let rec go acc = function
    | [] -> Ok (List.rev acc)
    | ((key : Id.Canonical_value.t), read) :: rest -> (
        match
          List.find_opt
            (fun (c : Canonical_spec.t) -> Id.Canonical_value.equal c.key key)
            mapping.frontier.canonical_values
        with
        | None -> go acc rest
        | Some _
          when List.exists (fun k -> Id.Canonical_value.equal k key) known ->
            go acc rest
        | Some c ->
            let* bytes = read () in
            let* v =
              Inference_compare_codecs.Logical_value.make ctx c.spec bytes
              |> Result.map_error (fun e ->
                  Format.asprintf "pinned %s: %a"
                    (Id.Canonical_value.to_string key)
                    Inference_compare_codecs.Codec_error.pp (Err.Error.kind e))
            in
            go ((key, v) :: acc) rest)
  in
  go [] pins
