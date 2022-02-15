#pragma once

#define WIDE_INTEGER_HAS_LIMB_TYPE_UINT64
#define WIDE_INTEGER_HAS_MUL_8_BY_8_UNROLL
#include "uintwide_t.h"

template<const math::wide_integer::size_t Width2,
    typename LimbType = std::uint32_t,
    typename AllocatorType = void,
    const bool IsSigned = false>
class adapter_uintwide_t : public math::wide_integer::uintwide_t<Width2, LimbType, AllocatorType, IsSigned> {
public:
    using super = math::wide_integer::uintwide_t<Width2, LimbType, AllocatorType, IsSigned>;
    using super::uintwide_t;

    adapter_uintwide_t(super s) {
        this->representation() = s.representation();
    }

    adapter_uintwide_t(std::string_view s) {
        constexpr auto size_l = sizeof(LimbType);
        std::fill(this->representation().begin(), this->representation().end(), LimbType(0U));
        auto out = reinterpret_cast<char *>(this->representation().data() + this->number_of_limbs) - 1;
        for (auto i = 0; i < std::min<size_t>(s.length(), this->number_of_limbs * size_l); ++i)
            *out-- = s[i];
    }

    // Different signedness and width cast
    template<const bool OtherIsSigned, const math::wide_integer::size_t OtherWidth2,
        typename std::enable_if<(Width2 != OtherWidth2)>::type const * = nullptr>
    adapter_uintwide_t(const adapter_uintwide_t<OtherWidth2, LimbType, AllocatorType, OtherIsSigned> &v) {
        super x = v;
        this->representation() = x.representation();
    }
};