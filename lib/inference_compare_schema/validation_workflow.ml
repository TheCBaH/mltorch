module W = Wire
module Case = W.Text (Id.Case)

type t = {
  reference_side : Side.t;
  tested_side : Side.t;
  contract_digest : Digest.Contract.t;
  cases : Id.Case.t list;
  partition_fidelity : Partition_fidelity.t;
  end_to_end_reference_check : bool;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* reference_side = W.Obj.required o "reference_side" Side.decode in
      let* tested_side = W.Obj.required o "tested_side" Side.decode in
      let* contract_digest =
        W.Obj.required o "contract_digest" Wire_digest.contract
      in
      let* cases = W.Obj.required o "cases" (W.list Case.decode) in
      let* partition_fidelity =
        W.Obj.required o "partition_fidelity" Partition_fidelity.decode
      in
      let* end_to_end_reference_check =
        W.Obj.required o "end_to_end_reference_check" W.bool
      in
      Ok
        {
          reference_side;
          tested_side;
          contract_digest;
          cases;
          partition_fidelity;
          end_to_end_reference_check;
        })

let encode w =
  W.obj
    [
      W.field "reference_side" Side.encode w.reference_side;
      W.field "tested_side" Side.encode w.tested_side;
      W.field "contract_digest" Wire_digest.encode_contract w.contract_digest;
      W.field "cases" (W.enc_list Case.encode) w.cases;
      W.field "partition_fidelity" Partition_fidelity.encode
        w.partition_fidelity;
      W.field "end_to_end_reference_check" W.enc_bool
        w.end_to_end_reference_check;
    ]
