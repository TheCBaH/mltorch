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

let identity ?(numerics = "scalar binary64") ~engine ~route ~components () =
  Ident.make ~engine ~route
    ~configuration:(numerics ^ "; isa=" ^ Isa.name)
    ~environment:
      (String.concat ":" [ Sys.os_type; string_of_int Sys.word_size; Isa.name ])
    ~components

let compcert_identity =
  identity ~engine:"compcert-embedded"
    ~route:("embedded-compcert-" ^ Isa.name)
    ~components:(components `Assembly)

let compcert_fp32_identity =
  identity ~numerics:"scalar binary32 (fp32_ordered)"
    ~engine:"compcert-embedded-fp32"
    ~route:("embedded-compcert-" ^ Isa.name)
    ~components:(components `Assembly)

let assembly_identity =
  identity ~engine:"rivet-assembly"
    ~route:("direct-assembly-" ^ Isa.name)
    ~components:(components `Compcert)

module Compcert = P.Graph_runner.Make (Compcert_engine)
module Assembly = P.Graph_runner.Make (Inference_compare_rivet.Image_engine)

module Compcert_fp32_engine = Compcert_engine.Make (struct
  let numerics = Loop_ir.Loop_numerics.Fp32_ordered
end)

module Compcert_fp32 = P.Graph_runner.Make (Compcert_fp32_engine)

let compcert () = Compcert.create ~identity:compcert_identity
let compcert_fp32 () = Compcert_fp32.create ~identity:compcert_fp32_identity
let assembly () = Assembly.create ~identity:assembly_identity
