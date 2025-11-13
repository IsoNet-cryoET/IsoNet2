import { STORE_KEYS } from '../constants.js'

export default function counter({ store }) {
    return {
        async getCurrentId() {
            return Number(store.get(STORE_KEYS.COUNTER, 0))
        },
        async nextId() {
            const next = Number(store.get(STORE_KEYS.COUNTER, 0)) + 1
            store.set(STORE_KEYS.COUNTER, next)
            return next
        },
    }
}
