open Err.Syntax

module Segment = struct
  type t = { align : int64; executable : bool; name : string; size : int64 }
end

module Placement = struct
  type t = In | Out | Scratch

  let to_string = function In -> "in" | Out -> "out" | Scratch -> "scratch"
end

module Slot = struct
  type t = { align : int64; placement : Placement.t; size : int64 }
end

module Params = struct
  type t = { stack : int64; staging : int64 }

  let default = { stack = 4096L; staging = 4096L }
end

module Placed_slot = struct
  type t = {
    address : int64;
    htif_reachable : bool;
    region : string;
    slot : Slot.t;
  }
end

type t = {
  digest : int32;
  htif : int64;
  packet : int64;
  segments : (string * int64) list;
  slots : Placed_slot.t list;
  stack_top : int64;
  staging : int64;
  staging_size : int64;
  trap : int64;
}

let htif_size = 256L
let packet_size = 64L
let trap_size = 128L

module Over_capacity = struct
  type t = { available : int64; needed : int64; region : string }
end

type error =
  [ `Missing_region of Coral_profile.Role.t | `Over_capacity of Over_capacity.t ]

let pp_error ppf : error -> unit = function
  | `Missing_region role ->
      Fmt.pf ppf "the profile has no usable %s region"
        (Coral_profile.Role.to_string role)
  | `Over_capacity { Over_capacity.available; needed; region } ->
      Fmt.pf ppf "%s: %Ld bytes needed, %Ld available" region needed available

let align_up n a =
  if a <= 1L then n
  else Int64.mul (Int64.div (Int64.add n (Int64.sub a 1L)) a) a

(* A cursor over one region, filling upwards to [limit]. *)
type cursor = { at : int64; limit : int64; name : string }

let take c ~align ~size =
  let start = align_up c.at align in
  let stop = Int64.add start size in
  if stop > c.limit then
    Err.fail
      (`Over_capacity
         {
           Over_capacity.available = Int64.max 0L (Int64.sub c.limit start);
           needed = size;
           region = c.name;
         })
  else Err.return (start, { c with at = stop })

let fits c ~align ~size = Int64.add (align_up c.at align) size <= c.limit

let region (p : Coral_profile.t) role ~writable =
  List.find_opt
    (fun (r : Coral_profile.Region.t) ->
      r.role = role && ((not writable) || r.writable))
    p.regions
  |> Err.of_option (`Missing_region role)

let cursor (r : Coral_profile.Region.t) =
  { at = r.base; limit = Int64.add r.base r.size; name = r.name }

let take_all c segments =
  Err.List.fold_left
    (fun (acc, c) (s : Segment.t) ->
      let* at, c = take c ~align:s.align ~size:s.size in
      Err.return ((s.name, at) :: acc, c))
    ([], c) segments
  |> Err.map (fun (acc, c) -> (List.rev acc, c))

(* The assignment as text, hashed for the digest. *)
let describe (p : Coral_profile.t) l =
  let b = Buffer.create 256 in
  Printf.bprintf b "%s\n" p.configuration_id;
  List.iter (fun (n, a) -> Printf.bprintf b "segment %s %Lx\n" n a) l.segments;
  Printf.bprintf b
    "htif %Lx\npacket %Lx\nstaging %Lx %Lx\ntrap %Lx\nstack %Lx\n" l.htif
    l.packet l.staging l.staging_size l.trap l.stack_top;
  List.iter
    (fun (s : Placed_slot.t) ->
      Printf.bprintf b "slot %s %Lx %Lx\n"
        (Placement.to_string s.slot.placement)
        s.address s.slot.size)
    l.slots;
  Buffer.contents b

let digest_of text =
  let h = Pt2_sha256.create () in
  Pt2_sha256.add_string h text;
  let hex = Pt2_sha256.Digest.to_hex (Pt2_sha256.finish h) in
  Int32.of_string ("0x" ^ String.sub hex 0 8)

let place ?(params = Params.default) (p : Coral_profile.t) segments slots =
  let* itcm = region p Itcm ~writable:false in
  let* dtcm = region p Dtcm ~writable:true in
  let code, data =
    List.partition (fun (s : Segment.t) -> s.executable) segments
  in
  let* code, _ = take_all (cursor itcm) code in
  let stack_top =
    Int64.logand (Int64.add dtcm.base dtcm.size) (Int64.lognot 15L)
  in
  let d = { (cursor dtcm) with limit = Int64.sub stack_top params.stack } in
  let* () =
    if d.limit < d.at then
      Err.fail
        (`Over_capacity
           {
             Over_capacity.available = dtcm.size;
             needed = params.stack;
             region = dtcm.name;
           })
    else Err.return ()
  in
  let* data, d = take_all d data in
  let* htif, d = take d ~align:64L ~size:htif_size in
  let* packet, d = take d ~align:8L ~size:packet_size in
  let* staging, d = take d ~align:16L ~size:params.staging in
  let* trap, d = take d ~align:16L ~size:trap_size in
  let ram = region p Ram ~writable:true in
  let* slots, _, _ =
    Err.List.fold_left
      (fun (acc, d, ext) (s : Slot.t) ->
        if fits d ~align:s.align ~size:s.size then
          let* address, d = take d ~align:s.align ~size:s.size in
          Err.return
            ( {
                Placed_slot.address;
                htif_reachable = dtcm.htif_reachable;
                region = dtcm.name;
                slot = s;
              }
              :: acc,
              d,
              ext )
        else
          let* r = ram in
          let e = Option.value ext ~default:(cursor r) in
          let* address, e = take e ~align:s.align ~size:s.size in
          Err.return
            ( {
                Placed_slot.address;
                htif_reachable = r.htif_reachable;
                region = r.name;
                slot = s;
              }
              :: acc,
              d,
              Some e ))
      ([], d, None) slots
  in
  let l =
    {
      digest = 0l;
      htif;
      packet;
      segments = code @ data;
      slots = List.rev slots;
      stack_top;
      staging;
      staging_size = params.staging;
      trap;
    }
  in
  Err.return { l with digest = digest_of (describe p l) }

let pp ppf l =
  let row ppf (name, at) = Fmt.pf ppf "%-10s 0x%08Lx" name at in
  Fmt.pf ppf "@[<v>digest %08lx@,%a@,%a@,%a@,%a@,%a@,%a%a@]" l.digest
    Fmt.(list row)
    l.segments row ("htif", l.htif) row ("packet", l.packet) row
    ("staging", l.staging) row ("trap", l.trap) row ("stack top", l.stack_top)
    Fmt.(
      list ~sep:nop (fun ppf (s : Placed_slot.t) ->
          Fmt.pf ppf "@,%-10s 0x%08Lx %Ld %s%s"
            ("slot " ^ Placement.to_string s.slot.placement)
            s.address s.slot.size s.region
            (if s.htif_reachable then "" else " staged")))
    l.slots
