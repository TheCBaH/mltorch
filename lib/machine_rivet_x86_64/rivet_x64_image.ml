(* A typed module through Rivet's lowering, relaxation and layout to a laid-out
   image, and a laid-out image bound at chosen addresses. This host cannot load
   x86-64 code, so the bound image goes to {!Rivet_x64_elf} instead of into this
   process. *)

module P = Driver_direct.Pipeline_direct.Make (X86_64_encode)

module Error = struct
  type t =
    [ `Rivet_bind of string | `Rivet_lower of string | `Rivet_plan of string ]

  let pp fmt : t -> unit = function
    | `Rivet_bind m | `Rivet_lower m | `Rivet_plan m -> Fmt.string fmt m
end

type laid_out = Image.laid_out

let render e = Foundation.Diag.render e

let plan ~entry modules : (laid_out, Error.t) Err.t =
  let rec lower acc = function
    | [] -> Ok (List.rev acc)
    | m :: rest -> (
        match P.lower ~state:X86_64_encode.default_state m with
        | Ok l -> lower (l :: acc) rest
        | Error e -> Err.fail (`Rivet_lower (render e)))
  in
  Err.bind (lower [] modules) (fun lowered ->
      match
        match lowered with
        | [ one ] -> P.plan ~entry one
        | many -> P.plan_many ~entry many
      with
      | Ok laid -> Err.return laid
      | Error e -> Err.fail (`Rivet_plan (render e)))

let page = 4096L
let align_up n a = Int64.mul (Int64.div (Int64.add n (Int64.sub a 1L)) a) a

(* The image at [base]: every segment on its own pages, in plan order. *)
let bind ~base (laid : laid_out) : (Image.t, Error.t) Err.t =
  let plan = Image.plan_of laid in
  let addresses, _ =
    List.fold_left
      (fun (acc, at) (s : Image.segment_plan) ->
        let at =
          align_up at (Int64.max page (Int64.of_int s.Image.alignment))
        in
        ( (s.Image.seg_name, at) :: acc,
          Int64.add at (Int64.of_int (s.Image.init_size + s.Image.zero_fill)) ))
      ([], base) plan.Image.segments
  in
  match Image.bind_image laid ~addresses:(List.rev addresses) with
  | Ok image -> Err.return image
  | Error e -> Err.fail (`Rivet_bind (render e))
