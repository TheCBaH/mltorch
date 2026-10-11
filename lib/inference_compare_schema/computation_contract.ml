module W = Wire
module Occurrence = W.Text (Id.Output_occurrence)

type t = {
  reference : Reference_role.t;
  input_domain : Input_domain.t;
  local_policy_selection : Policy_selection.t;
  final_policies : (Id.Output_occurrence.t * Comparison_policy.t) list;
  allowed_conversion_digest : Digest.Conversion.t list;
  required_full_graph_acceptance : bool;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* reference = W.Obj.required o "reference" Reference_role.decode in
      let* input_domain = W.Obj.required o "input_domain" Input_domain.decode in
      let* local_policy_selection =
        W.Obj.required o "local_policy_selection" Policy_selection.decode
      in
      let* final_policies =
        W.Obj.required o "final_policies"
          (W.list (W.pair Occurrence.decode Comparison_policy.decode))
      in
      let* allowed_conversion_digest =
        W.Obj.required o "allowed_conversion_digest"
          (W.list Wire_digest.conversion)
      in
      let* required_full_graph_acceptance =
        W.Obj.required o "required_full_graph_acceptance" W.bool
      in
      Ok
        {
          reference;
          input_domain;
          local_policy_selection;
          final_policies;
          allowed_conversion_digest;
          required_full_graph_acceptance;
        })

let encode c =
  W.obj
    [
      W.field "reference" Reference_role.encode c.reference;
      W.field "input_domain" Input_domain.encode c.input_domain;
      W.field "local_policy_selection" Policy_selection.encode
        c.local_policy_selection;
      W.field "final_policies"
        (W.enc_list (W.enc_pair Occurrence.encode Comparison_policy.encode))
        c.final_policies;
      W.field "allowed_conversion_digest"
        (W.enc_list Wire_digest.encode_conversion)
        c.allowed_conversion_digest;
      W.field "required_full_graph_acceptance" W.enc_bool
        c.required_full_graph_acceptance;
    ]

let canonicalize c =
  {
    c with
    local_policy_selection =
      Policy_selection.canonicalize c.local_policy_selection;
    final_policies =
      List.stable_sort
        (fun (a, _) (b, _) -> Id.Output_occurrence.compare a b)
        (List.map
           (fun (o, p) -> (o, Comparison_policy.canonicalize p))
           c.final_policies);
    allowed_conversion_digest =
      List.sort_uniq Digest.Conversion.compare c.allowed_conversion_digest;
  }

let preimage c = Digest_preimage.of_json Contract (encode (canonicalize c))
