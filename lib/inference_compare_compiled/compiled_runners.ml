(* The two compiled routes as ordinary graph runners over Native bodies, each
   with an identity naming its compiler and loader. The ISA is the one the
   embedding was built for. *)

module P = Inference_compare_pt2
module Ident = Inference_compare_rivet.Compiled_identity

let components routes =
  List.filter_map
    (fun (c : Ancestry.component) ->
      if c.routes <> routes then Some { Ident.name = c.name; pin = c.pin }
      else None)
    Ancestry.components

let identity ~engine ~route ~components () =
  Ident.make ~engine ~route
    ~configuration:("scalar binary64; isa=" ^ Isa.name)
    ~environment:
      (String.concat ":" [ Sys.os_type; string_of_int Sys.word_size; Isa.name ])
    ~components

let compcert_identity =
  identity ~engine:"compcert-embedded"
    ~route:("embedded-compcert-" ^ Isa.name)
    ~components:(components `Assembly)

let assembly_identity =
  identity ~engine:"rivet-assembly"
    ~route:("direct-assembly-" ^ Isa.name)
    ~components:(components `Compcert)

module Compcert = P.Graph_runner.Make (Compcert_engine)
module Assembly = P.Graph_runner.Make (Inference_compare_rivet.Image_engine)

let compcert () = Compcert.create ~identity:compcert_identity
let assembly () = Assembly.create ~identity:assembly_identity
