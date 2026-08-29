(* Ceilings and profiles for the Model Explorer export path. See the .mli for
   what the three layers are and why they are three.

   This file is now a thin facade: error payload types live in
   me_limits_error.ml, the Scope/Field/Over_limit domain in
   me_limits_over_limit.ml, the checked arithmetic and Hard ceilings in
   me_limits_hard.ml, the Limits profile in me_limits_profile.ml, the wire
   subset in me_limits_wire.ml, and Diagnostic in me_limits_diagnostic.ml.
   me_limits.mli is unchanged by the split: every item below is a manifest
   alias, so the public [Me_limits.*] surface is exactly what it was before.. *)

module Invalid = Me_limits_error.Invalid

type live_error = Me_limits_error.live_error
type error = Me_limits_error.error

let pp_error = Me_limits_error.pp_error

module Scope = Me_limits_over_limit.Scope
module Field = Me_limits_over_limit.Field
module Over_limit = Me_limits_over_limit.Over_limit

type over_limit_error = Me_limits_over_limit.over_limit_error

let check = Me_limits_over_limit.check
let check64 = Me_limits_over_limit.check64

module Hard = Me_limits_hard.Hard

let response_live_bytes = Me_limits_hard.response_live_bytes

module Limits = Me_limits_profile.Limits
module Wire_limits = Me_limits_wire.Wire_limits
module Diagnostic = Me_limits_diagnostic.Diagnostic
