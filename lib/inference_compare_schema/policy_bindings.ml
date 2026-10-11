module W = Wire

type t = {
  selection_digest : Digest.Policy_selection.t;
  bindings_digest : Digest.Policy_bindings.t;
  bindings : Policy_binding.t list;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* selection_digest =
        W.Obj.required o "selection_digest" Wire_digest.policy_selection
      in
      let* bindings_digest =
        W.Obj.required o "bindings_digest" Wire_digest.policy_bindings
      in
      let* bindings =
        W.Obj.required o "bindings" (W.list Policy_binding.decode)
      in
      Ok { selection_digest; bindings_digest; bindings })

let encode b =
  W.obj
    [
      W.field "selection_digest" Wire_digest.encode_policy_selection
        b.selection_digest;
      W.field "bindings_digest" Wire_digest.encode_policy_bindings
        b.bindings_digest;
      W.field "bindings" (W.enc_list Policy_binding.encode) b.bindings;
    ]

(* Canonical form: bindings by comparison id, matches by rule id. A site's
   policy is already in canonical metric order. The bindings digest is not part
   of its own preimage. *)
let canonicalize t =
  let sort_by key items =
    List.stable_sort (fun a b -> String.compare (key a) (key b)) items
  in
  let binding (b : Policy_binding.t) =
    {
      b with
      Policy_binding.policy = Comparison_policy.canonicalize b.policy;
      matches =
        sort_by
          (fun (m : Policy_binding.Rule_match.t) ->
            Id.Metric_rule.to_string m.rule)
          b.matches;
      base =
        (match b.base with
        | Rules rs -> Rules (List.sort Id.Metric_rule.compare rs)
        | (Default | Explicit) as x -> x);
    }
  in
  {
    t with
    bindings =
      sort_by
        (fun (b : Policy_binding.t) ->
          Id.Comparison.to_string b.site.comparison)
        (List.map binding t.bindings);
  }

let preimage t =
  let t = canonicalize t in
  Digest_preimage.of_json Policy_bindings
    (W.obj
       [
         W.field "selection_digest" Wire_digest.encode_policy_selection
           t.selection_digest;
         W.field "bindings" (W.enc_list Policy_binding.encode) t.bindings;
       ])
