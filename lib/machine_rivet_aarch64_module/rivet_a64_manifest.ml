(* What identifies and declares a published image: the target and features it
   needs, the numerical planning it was made under, how its regions are bound,
   the helpers it depends on and the mode that admitted them, the helpers it
   carries its own code for (by digest), and a digest of its code. Two artifacts with equal manifests are interchangeable; a cache
   keyed by {!key} cannot confuse a binary32 plan with a binary64 one, a
   dependency-free image with a libm one, or one allocator's code with
   another's. The digest is over the printed module, so it is the same in every
   process: a helper's stub carries a process-specific address and is not part
   of it. *)

open Machine_ir
module Art = Machine_model.Mir_artifact
module T = Rivet_a64_table

type t = {
  target : string;
  source : string;  (** the architecture document and revision the forms cite *)
  features : string list;
  planning : string;  (** {!Mir_planning.to_string}, canonical *)
  runtime : string;
  binding : string;
  helpers : string list;
  owned : (string * string) list;
      (** helpers the image carries, with a digest of that code *)
  regions : (string * int64 * int64 * string) list;
      (** symbol, bytes, alignment, where the bytes come from *)
  code_digest : string;  (** MD5, hex, of the artifact's GNU source *)
}

let binding_name = function
  | Rivet_a64_module.Image_resident -> "image_resident"
  | Rivet_a64_module.Table -> "table"

let digest_of modul =
  Digest.to_hex
    (Digest.string
       (Fmt.str "%a"
          (Asm_core.Gnu_module.pp
             { Asm_core.Gnu_module.type_char = '%' }
             ~instruction:Aarch64_encode.Instruction.pp_gnu)
          modul))

(* The owned helper's code, the same in every process. *)
let owned_digest = function
  | "exp" -> (
      match Err.payload Rivet_a64_exp.module_ with
      | Ok m -> digest_of m
      | Error r ->
          invalid_arg (Fmt.str "Rivet_a64_manifest: %a" Rivet_a64_refusal.pp r))
  | h -> invalid_arg ("Rivet_a64_manifest: no owned helper " ^ h)

let make ~runtime ~binding artifact =
  let id = Art.identity artifact in
  let modul =
    match Err.payload (Rivet_a64_module.of_artifact ~binding artifact) with
    | Ok m -> m
    | Error r ->
        invalid_arg (Fmt.str "Rivet_a64_manifest: %a" Rivet_a64_refusal.pp r)
  in
  {
    target = id.Art.Identity.target;
    source =
      Fmt.str "%s %s" id.Art.Identity.source.Mir_target.Source.document
        id.Art.Identity.source.Mir_target.Source.revision;
    features = List.map Mir_target.Feature.name id.Art.Identity.features;
    planning = Mir_planning.to_string id.Art.Identity.planning;
    runtime = Rivet_a64_runtime.name runtime;
    binding = binding_name binding;
    helpers = Rivet_a64_runtime.bound runtime artifact;
    owned =
      List.map
        (fun h -> (h, owned_digest h))
        (Rivet_a64_runtime.carried runtime artifact);
    regions =
      List.map
        (fun s ->
          (s.T.symbol, s.T.size, s.T.align, Art.Section.name s.T.section))
        (T.of_artifact artifact);
    code_digest =
      Digest.to_hex
        (Digest.string
           (Fmt.str "%a"
              (Asm_core.Gnu_module.pp
                 { Asm_core.Gnu_module.type_char = '%' }
                 ~instruction:Aarch64_encode.Instruction.pp_gnu)
              modul));
  }

(* Canonical text: one [key=value] line per field. *)
let to_string t =
  String.concat "\n"
    ([
       "target=" ^ t.target;
       "source=" ^ t.source;
       "features=" ^ String.concat "," t.features;
       "runtime=" ^ t.runtime;
       "binding=" ^ t.binding;
       "helpers=" ^ String.concat "," t.helpers;
       "owned="
       ^ String.concat "," (List.map (fun (h, d) -> h ^ ":" ^ d) t.owned);
       "code=" ^ t.code_digest;
     ]
    @ List.map
        (fun (n, size, align, section) ->
          Fmt.str "region=%s,%Ld,%Ld,%s" n size align section)
        t.regions
    @ List.map (fun l -> "planning." ^ l) (String.split_on_char '\n' t.planning)
    )
  ^ "\n"

let equal a b = String.equal (to_string a) (to_string b)
let key t = Digest.to_hex (Digest.string (to_string t))
