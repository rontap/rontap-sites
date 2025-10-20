import {Consumables} from "../powerups/Consumables.ts";
import Card from "../components/card/Card.tsx";
import {Consumable} from "../components/Powerup.tsx";
import {Run} from "../Run.ts";

type CardShopProps = {
    run: Run
}
export default function CardShop(props: CardShopProps) {
    return <>
        <div className={"shopCardSelection"}>
            {
                Object.values(Consumables).map((el: Consumable) => {
                    return <>
                        <Card icon={el.self.icon} onClick={() => props.run.acquirePowerup(el)}
                              title={el.self.name}
                              standalone={true} total={0}
                              cost={el.self.cost}
                              nth={0}
                              toggle={() => false} selected={false}
                              bg={el.self.bg || "#666"}
                              kind={String(el.self.use)}
                              standablone={true}
                              faction={el.self.factions}>
                            {el.self.description}
                        </Card>
                    </>
                })
            }
        </div>
    </>
}