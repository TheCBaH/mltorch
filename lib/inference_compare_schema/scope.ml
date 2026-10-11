module W = Wire

type t = Name.Scope_component.t list

module C = W.Text (Name.Scope_component)

let compare = List.compare Name.Scope_component.compare
let equal a b = compare a b = 0
let to_strings = List.map Name.Scope_component.to_string
let pp = Fmt.(list ~sep:(any "/") Name.Scope_component.pp)
let decode : t W.decoder = W.list C.decode
let encode : t W.encoder = W.enc_list C.encode
