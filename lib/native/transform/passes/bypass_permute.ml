(* Remove individual inverse consumers of a permute, keeping a shared producer
   alive when needed instead of requiring the whole run to be interior. See
   .ai/native_layout_reuse_plan.md and .ai/native_transform_design.md §12a's
   [Trim_permute], which this complements:

     x -> P -> y -> Q -> z    becomes    x ----------------> z
                                           \-> P -> y

   [Trim_permute] cancels a COMPLETE identity run and therefore requires every
   intermediate edge it deletes to be interior; [Bypass_permute] instead
   removes one or more inverse consumers of a producer that may have OTHER
   consumers or be a graph output, in which case the producer itself has to
   stay. *)

open Graph_ir

let as_permute = function
  | Permute (p : Permute.Permute.t) -> Some p
  | _ -> None

(* [packed_fmt] is an existential over a GADT and has no usable structural
   order, so compare by format name, same as [Correspondence.Precision]. *)
let precision_equal (a : Tensor_sig.t) (b : Tensor_sig.t) =
  let fmt_key (Payload.Fmt f) = Payload.fmt_name f in
  String.equal (fmt_key a.Tensor_sig.fmt) (fmt_key b.Tensor_sig.fmt)
  && Stdlib.( = ) a.Tensor_sig.quant b.Tensor_sig.quant

(* One collected inverse consumer: the [Permute(Q)] node to remove and its
   output edge, which every one of its own consumers will read [x] instead
   of. Keeps the whole [node], not just its id, so it can be [claim]ed
   directly rather than reconstructed. *)
module Inverse_consumer = struct
  type t = { node : node; out : Tensor_id.t }
end

module Match = struct
  type t = {
    p_node : Node_id.t; (* claimed only when [remove_p] *)
    x : Tensor_id.t; (* P's input: what every [z] becomes *)
    y : Tensor_id.t; (* the anchor: P's output *)
    remove_p : bool;
    consumers : Inverse_consumer.t list;
  }
end

let pattern anchor =
  let open Pattern in
  let* (p : Permute.Permute.t), (p_node : node) = peek anchor as_permute in
  let* x_sig = sig_of p.x in
  let* consumers = uses anchor in
  let inverse_consumers =
    List.filter_map
      (fun (n : node) ->
        match (as_permute n.Node.op, n.Node.outputs) with
        | Some (q : Permute.Permute.t), [ out ]
          when Permute.Permute.are_inverse ~before:p.perm ~after:q.perm ->
            Some { Inverse_consumer.node = n; out }
        | _ -> None)
      consumers
  in
  (* Filter, not guard-all: an incompatible inverse consumer is not bypassed,
     but it must not sink a match where OTHER consumers are perfectly
     bypassable — it simply counts against [all_covered] below, keeping [P]
     live for its sake, exactly like any other non-inverse consumer. *)
  let* inverse_consumers =
    let+ compatible =
      List.fold_left
        (fun acc (c : Inverse_consumer.t) ->
          let* acc = acc in
          let+ out_sig = sig_of c.out in
          if precision_equal x_sig out_sig then c :: acc else acc)
        (return []) inverse_consumers
    in
    List.rev compatible
  in
  let* () = guard (inverse_consumers <> []) in
  (* Claimed atomically, in one match, so greedy scanning cannot split the
     fan-out into overlapping recipes. *)
  let* () =
    List.fold_left
      (fun acc (c : Inverse_consumer.t) ->
        let* () = acc in
        claim c.Inverse_consumer.node)
      (return ()) inverse_consumers
  in
  let all_covered = List.length consumers = List.length inverse_consumers in
  let* is_output = is_graph_output anchor in
  let remove_p = all_covered && not is_output in
  let+ () = if remove_p then claim p_node else return () in
  {
    Match.p_node = p_node.Node.id;
    x = p.x;
    y = anchor;
    remove_p;
    consumers = inverse_consumers;
  }

let build (m : Match.t) _region =
  let open Recipe in
  let tie =
    List.map (fun (c : Inverse_consumer.t) -> (c.out, m.Match.x)) m.consumers
  in
  let remove =
    (if m.Match.remove_p then [ m.Match.p_node ] else [])
    @ List.map (fun (c : Inverse_consumer.t) -> c.node.Node.id) m.consumers
  in
  trim ~remove ~tie

let pass = Pass.of_pattern ~name:"bypass_permute" ~pattern ~build
