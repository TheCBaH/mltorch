module W = Wire

type t = Bitwise_required | Declared_contract

let decode, encode =
  W.enum ~kind:"partition_fidelity"
    [
      (Bitwise_required, "bitwise_required");
      (Declared_contract, "declared_contract");
    ]
